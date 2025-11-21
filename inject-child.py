from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.storage import InMemoryStore
from langchain_core.documents import Document

# Documents parents exemples
docs = [
    Document(page_content="LangChain est un framework pour développer des applications basées sur des modèles de langage."),
    Document(page_content="Il permet des applications conscientes du contexte et capables de raisonner sur des documents."),
    Document(page_content="Le parent-child chunking aide à récupérer des documents complets tout en indexant des parties plus petites."),
]

# Splitters
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

# Embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Création du vectorstore pour les child chunks
child_docs = child_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=child_docs, embedding=embeddings)

# Stockage en mémoire pour les parents
docstore = InMemoryStore()

# Retriever parent-child
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# Ajout des documents parents
retriever.add_documents(docs)

# Exemple de query
query = "Qu'est-ce que LangChain ?"
results = retriever.invoke(query)

# Affichage des résultats (parents contenant les child matchés)
for i, doc in enumerate(results, 1):
    print(f"\n[Résultat {i}]\n{doc.page_content}")