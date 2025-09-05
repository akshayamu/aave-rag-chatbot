import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def load_and_process_documents(data_dir="data"):
    documents = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(data_dir, filename), encoding="utf-8")
            documents.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50, length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    return splitter.split_documents(documents)

def create_vectorstore(documents, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return FAISS.from_documents(documents, embeddings)

def save_vectorstore(vectorstore, path="models/aave_faiss_index"):
    os.makedirs(path, exist_ok=True)
    vectorstore.save_local(path)

def load_vectorstore(path="models/aave_faiss_index", model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
