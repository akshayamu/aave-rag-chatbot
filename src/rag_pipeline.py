import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


class AaveRAGPipeline:
    def __init__(self, data_folder: str = "data"):
        # HuggingFace embeddings (cloud-friendly)
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # LLM (Groq)
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.1-8b-instant"
        )

        # Internal vars
        self.vectorstore = None
        self.qa_chain = None
        self.data_folder = data_folder

    def load_and_index(self):
        """Load PDFs from data/ and build FAISS vectorstore"""
        documents = []
        for file in os.listdir(self.data_folder):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(self.data_folder, file))
                docs = loader.load()
                documents.extend(docs)

        if not documents:
            raise ValueError("❌ No documents found in data/ folder.")

        # Split documents
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(documents)

        # Build FAISS vectorstore
        self.vectorstore = FAISS.from_documents(splits, self.embeddings)

        # Build QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True
        )

    def ask(self, query: str):
        """Answer user queries with sources"""
        if not self.qa_chain:
            raise ValueError("⚠️ Vectorstore not built. Run load_and_index() first.")

        result = self.qa_chain(query)
        answer = result["result"]
        sources = [doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]
        return answer, sources
