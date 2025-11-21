#!/bin/python3

import os
import logging

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
import chromadb

model_embeddings = 'qwen3-embedding:8b'
chroma_host = 'localhost'
chroma_port = 8000
collection_name = 'docs'
data_dir = './data'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""])

embeddings = OllamaEmbeddings(model=model_embeddings)
client = chromadb.HttpClient(host=chroma_host, port=chroma_port)

try:
    client.delete_collection(name=collection_name)
    logger.info(f"Deleted existing collection: {collection_name}")
except Exception as e: pass

vectorstore = Chroma(client=client, embedding_function=embeddings, collection_name=collection_name)

documents = []
for root, dirs, files in os.walk(data_dir):
    for file in files:
        file_path = os.path.join(root, file)
        try:
            logger.info(f"Processing file: {file}")
            documents.extend(splitter.split_documents(TextLoader(file_path).load()))
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")

logger.info("Inject %s documents in vector" % len(documents))
vectorstore.add_documents(documents)
logger.info("Inject with success")
