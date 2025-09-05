# src/rag_pipeline.py
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_groq import ChatGroq

from config import GROQ_API_KEY, OLLAMA_EMBEDDING_MODEL, GROQ_LLM_MODEL
from document_processor import load_vectorstore


class AaveRAGPipeline:
    def __init__(self, model_path="models/aave_faiss_index"):
        """Initialize embeddings, vectorstore, retriever, and Groq LLM."""
        # 1. Embeddings (Ollama)
        self.embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)

        # 2. Vectorstore
        self.vectorstore = load_vectorstore(
            path=model_path,
            model_name=OLLAMA_EMBEDDING_MODEL
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        # 3. Groq LLM (dynamic model from config)
        if not GROQ_API_KEY:
            raise ValueError("❌ GROQ_API_KEY is missing. Please set it in your .env file.")

        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model=GROQ_LLM_MODEL,   # <- pulled from config.py
            temperature=0,
            max_tokens=512,
        )

        # 4. Prompt template
        self.prompt = PromptTemplate(
            template=(
                "You are an assistant for answering questions about Aave Protocol.\n"
                "Use the provided context to give accurate answers.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n"
                "Answer in detail:"
            ),
            input_variables=["context", "question"]
        )

        # 5. Build RetrievalQA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=True
        )

    def ask(self, question: str):
        """Query the pipeline and return answer, confidence, and sources."""
        if not question.strip():
            return {"answer": "⚠️ Please enter a valid question.", "confidence": 0, "source_documents": []}

        result = self.qa_chain({"query": question})

        # Extract fields safely
        answer = result.get("result", "❌ No answer generated.")
        sources = result.get("source_documents", [])

        # Basic confidence heuristic: if sources retrieved, confidence is higher
        confidence = 0.8 if sources else 0.4

        return {
            "answer": answer,
            "confidence": confidence,
            "source_documents": sources,
        }
