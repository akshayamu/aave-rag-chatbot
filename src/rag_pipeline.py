import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


class AaveRAGPipeline:
    def __init__(self, data_path="data"):
        self.data_path = data_path
        self.vectorstore = None
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.1-8b-instant"
        )
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def load_documents(self):
        """Load PDFs and create vectorstore"""
        documents = []
        if not os.path.exists(self.data_path):
            print("⚠️ Data folder not found!")
            return

        for file in os.listdir(self.data_path):
            if file.endswith(".pdf"):
                loader = PyPDFLoader(os.path.join(self.data_path, file))
                docs = loader.load()
                documents.extend(docs)

        if not documents:
            print("⚠️ No documents found in data folder.")
            return

        # Split text
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        # Build FAISS
        self.vectorstore = FAISS.from_documents(chunks, self.embedding_model)
        print(f"✅ Loaded {len(chunks)} chunks into vectorstore")

    def query(self, question, k=3):
        """Retrieve and answer with confidence"""
        if not self.vectorstore:
            return "No documents loaded. Please add PDFs to the data/ folder and restart the app.", 0.0

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(question)

        context = "\n\n".join([d.page_content for d in docs])
        prompt = f"""You are an Aave expert.
Context:
{context}

Question: {question}

Answer clearly and concisely.
"""
        response = self.llm.invoke(prompt)

        # Dummy confidence (based on number of docs retrieved)
        confidence = min(1.0, 0.6 + 0.1 * len(docs))

        return response.content, confidence
