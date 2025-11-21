#!/bin/python3

import logging

from langchain_ollama import OllamaEmbeddings
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
import chromadb

model_embeddings = 'qwen3-embedding:8b'
model_llm = 'gpt-oss:20b'
chroma_host = 'localhost'
chroma_port = 8000
collection_name = 'docs'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

embeddings = OllamaEmbeddings(model=model_embeddings)
client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
vectorstore = Chroma(client=client, embedding_function=embeddings, collection_name=collection_name)

qa = RetrievalQA.from_chain_type(llm=OllamaLLM(model=model_llm), chain_type="stuff", retriever=vectorstore.as_retriever())
q = input("Question : ")
logger.info("\n" + qa.invoke({"query": q})["result"])

# prompt_template = """
# Use the following pieces of context to answer the question at the end.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.
# Provide the answer in a structured format if possible.

# Context: {context}

# Question: {question}

# Answer:
# """
# PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# qa = RetrievalQA.from_chain_type(llm=OllamaLLM(model=model_llm), chain_type="stuff", retriever=vectorstore.as_retriever(), return_source_documents=True, chain_type_kwargs={"prompt": PROMPT})
# q = input("Question: ").strip()

# result = qa.invoke({"query": q})
# print("\nAnswer:", result["result"])
# print("Sources:", [doc.metadata.get('source', 'unknown') for doc in result["source_documents"]])

# Donne moi les attributs de la classe fakeServer sous la forme d'un tableau, aucune autre information ou commentaire n'est attendu. Ensuite dit comment je m'appelle
# Donne moi les attributs de la méthode __init__ de la classe Bot sous la forme d'un tableau, aucune autre information ou commentaire n'est attendu.
