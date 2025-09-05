import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq


class AaveRAGPipeline:
    def __init__(self):
        # ✅ Initialize embeddings (HuggingFace instead of Ollama)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # ✅ Load documents from data/ folder
        data_path = os.path.join(os.path.dirname(__file__), "..", "data")
        pdf_files = [f for f in os.listdir(data_path) if f.endswith(".pdf")]

        documents = []
        for pdf in pdf_files:
            loader = PyPDFLoader(os.path.join(data_path, pdf))
            documents.extend(loader.load())

        if not documents:
            raise ValueError("❌ No documents loaded. Please add PDFs to the data/ folder.")

        # ✅ Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        splits = text_splitter.split_documents(documents)

        # ✅ Build vectorstore
        self.vectorstore = FAISS.from_documents(splits, embeddings)

        # ✅ Groq LLM
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.1-8b-instant",
        )

    def query(self, question: str) -> str:
        """Main RAG query pipeline"""
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(question)

        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = f"""You are an assistant for Aave documentation.
Use the context below to answer the question.
If you are not confident, say you are not sure.

Context:
{context}

Question:
{question}

Answer:
"""

        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    def ask(self, question: str) -> str:
        """Alias for backwards compatibility"""
        return self.query(question)
