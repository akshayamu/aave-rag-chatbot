import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

def load_and_process_documents(data_dir="data"):
    """Load all .txt documents and split into chunks"""
    documents = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(data_dir, filename)
            loader = TextLoader(file_path, encoding="utf-8")
            documents.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return text_splitter.split_documents(documents)

def create_vectorstore(documents, model_name="all-minilm"):
    embeddings = OllamaEmbeddings(model=model_name)
    return FAISS.from_documents(documents, embeddings)

def save_vectorstore(vectorstore, path="models/aave_faiss_index"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vectorstore.save_local(path)

def load_vectorstore(path="models/aave_faiss_index", model_name="all-minilm"):
    embeddings = OllamaEmbeddings(model=model_name)
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
