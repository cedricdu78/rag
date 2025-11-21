from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from nltk import sent_tokenize
import nltk

import os
import logging

model_llm = 'gpt-oss:20b'
data_dir = './data'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Téléchargement NLTK pour tokenisation (optionnel, si besoin manuel)
nltk.download('punkt')

# Modèle d'embeddings (Sentence Transformers-like)
embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Chargement d'un document exemple
documents = []
for root, dirs, files in os.walk(data_dir):
    for file in files:
        file_path = os.path.join(root, file)
        try:
            logger.info(f"Processing file: {file}")
            documents.extend(TextLoader(file_path).load())
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")

# Option : Tokenisation manuelle avec NLTK si besoin
# text = documents[0].page_content
# sentences = sent_tokenize(text)  # Split en phrases

# Chunking sémantique automatique avec LangChain
semantic_chunker = SemanticChunker(
    embed_model,
    breakpoint_threshold_type="percentile"  # Ou "standard_deviation", "interquartile"
)

# Création des chunks sémantiques
semantic_chunks = semantic_chunker.create_documents([d.page_content for d in documents])

# Pour stocker dans un vectorstore (ex. Chroma)
vectorstore = Chroma.from_documents(semantic_chunks, embedding=embed_model)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

qa = RetrievalQA.from_chain_type(llm=OllamaLLM(model=model_llm), chain_type="stuff", retriever=retriever)
q = input("Question : ")
logger.info("\n" + qa.invoke({"query": q})["result"])
