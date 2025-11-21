#!/bin/python3

import os
import logging
import json

from langchain_ollama import ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling
from trl import SFTTrainer
import torch

# Note: You need to install the following packages if not already installed:
# pip install transformers datasets peft trl accelerate
# Note: bitsandbytes is not required or supported on Apple Silicon, so we're removing quantization
# If you see warnings from bitsandbytes, uninstall it: pip uninstall bitsandbytes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

model_llm = 'qwen2:7b'
base_model_name = 'Qwen/Qwen2.5-1.5B-Instruct'
data_dir = './data'

# llm = ChatOllama(model=model_llm)

# documents = []
# for root, dirs, files in os.walk(data_dir):
#     for file in files:
#         file_path = os.path.join(root, file)
#         try:
#             logger.info(f"Processing file: {file}")
#             documents.extend(splitter.split_documents(TextLoader(file_path).load()))
#         except Exception as e:
#             logger.error(f"Error loading {file}: {e}")

# logger.info(f"Loaded {len(documents)} document chunks")

# # Generate Q&A pairs for instruction fine-tuning
# dataset = []
# for i, doc in enumerate(documents):
#     chunk = doc.page_content
#     prompt = f"""Based on the following text, generate 3 question-answer pairs in JSON format like this: [{{"question": "q1", "answer": "a1"}}, ...]

# Do not include any additional text outside the JSON array.

# Text: {chunk}"""
#     try:
#         response = llm.invoke(prompt).content
#         pairs = json.loads(response)
#         for pair in pairs:
#             dataset.append({
#                 "instruction": pair["question"],
#                 "input": "",
#                 "output": pair["answer"]
#             })
#         logger.info(f"Generated pairs for chunk {i+1}/{len(documents)}")
#     except Exception as e:
#         logger.error(f"Error generating pairs for chunk {i+1}: {e}")

# logger.info(f"Generated {len(dataset)} instruction examples")

# Save dataset to JSONL
dataset_file = 'dataset.jsonl'
# with open(dataset_file, 'w') as f:
#     for item in dataset:
#         f.write(json.dumps(item) + '\n')

# Load dataset
ds = load_dataset('json', data_files=dataset_file)

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# Format function for Alpaca-style
def format_examples(examples):
    texts = []
    for instruction, input_text, output in zip(examples['instruction'], examples['input'], examples['output']):
        text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
        texts.append(text)
    return {"text": texts}

ds = ds.map(format_examples, batched=True, remove_columns=ds['train'].column_names)

# Load model without quantization
model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map="mps" if torch.backends.mps.is_available() else "cpu",
    dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32,
)

# LoRA config (adjust target_modules based on model architecture)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training arguments (adjust based on your hardware)
training_args = TrainingArguments(
    output_dir="./fine_tuned_results",
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Adjust based on GPU memory
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True if torch.backends.mps.is_available() else False,
    bf16=False,  # Set to True if your hardware supports bfloat16
    save_steps=500,
    logging_steps=100,
    optim="adamw_torch",
    weight_decay=0.01,
    warmup_steps=100,
    report_to="none",
)

# Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=ds['train'],
    args=training_args,
    processing_class=tokenizer,
    data_collator=data_collator,
)

logger.info("Starting fine-tuning...")
trainer.train()

# Merge and save
logger.info("Merging and saving the fine-tuned model")
peft_model = trainer.model
merged_model = peft_model.merge_and_unload()
merged_model.save_pretrained("merged_model")
tokenizer.save_pretrained("merged_model")

logger.info("Fine-tuning completed. Model saved to 'merged_model'.")

logger.info("""
To use the fine-tuned model with Ollama:
1. Install llama.cpp if not already: git clone https://github.com/ggerganov/llama.cpp
2. cd llama.cpp && make
3. python convert_hf_to_gguf.py ../merged_model --outfile ../mymodel.gguf --outtype f16  # Adjust outtype as needed
4. Create a Modelfile in the parent directory with content:
   FROM ./mymodel.gguf
5. ollama create my_fine_tuned_model -f Modelfile
6. Now you can run: ollama run my_fine_tuned_model
7. Ask questions like: "What is [something from your data]?"
Note: For better QA performance, use a prompt template matching the instruct model's format.
""")