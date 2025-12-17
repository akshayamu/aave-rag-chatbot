import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# ======================================================
# Environment
# ======================================================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_LLM_MODEL = os.getenv("GROQ_LLM_MODEL", "llama-3.1-8b-instant")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

# ======================================================
# Helper: lexical grounding score
# ======================================================
def grounding_score(answer: str, docs) -> float:
    if not docs or not answer:
        return 0.0

    context = " ".join(d.page_content.lower() for d in docs)
    tokens = [t for t in answer.lower().split() if len(t) > 3]

    if not tokens:
        return 0.0

    hits = sum(1 for t in tokens if t in context)
    return min(hits / len(tokens), 1.0)

# ======================================================
# Main RAG Pipeline
# ======================================================
class AaveRAGPipeline:
    """
    Conservative, evaluation-driven RAG pipeline
    """

    def __init__(self, k: int = 5):
        base_dir = Path(__file__).resolve().parents[1]

        self.data_dir = base_dir / "data"
        self.index_dir = base_dir / "models" / "aave_faiss_index"
        self.k = k

        print(f"[RAG] Data directory: {self.data_dir}")
        print(f"[RAG] Index directory: {self.index_dir}")

        # LLM
        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_LLM_MODEL,
            temperature=0.0
        )

        # Embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Vector store
        self.vectorstore = self._load_or_build_index()
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.k}
        )

    # --------------------------------------------------
    # Indexing
    # --------------------------------------------------
    def _load_or_build_index(self):
        if self.index_dir.exists():
            vs = FAISS.load_local(
                str(self.index_dir),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print(f"[RAG] Loaded FAISS index ({vs.index.ntotal} vectors)")
            return vs

        docs = self._load_documents()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)

        vs = FAISS.from_documents(chunks, self.embeddings)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        vs.save_local(str(self.index_dir))

        print(f"[RAG] Built FAISS index ({vs.index.ntotal} vectors)")
        return vs

    def _load_documents(self):
        if not self.data_dir.exists():
            raise RuntimeError("data/ directory missing")

        documents = []
        print("[RAG] Loading documents…")

        for file in self.data_dir.iterdir():
            if file.suffix.lower() == ".pdf":
                docs = PyPDFLoader(str(file)).load()
            elif file.suffix.lower() == ".txt":
                docs = TextLoader(str(file), encoding="utf-8").load()
            else:
                continue

            for d in docs:
                text = d.page_content.strip()
                if len(text) < 50:
                    continue

                d.metadata["source"] = file.name
                documents.append(d)

        if not documents:
            raise RuntimeError("No valid documents found")

        print(f"[RAG] Loaded {len(documents)} chunks")
        return documents

    # --------------------------------------------------
    # Query
    # --------------------------------------------------
    def query(self, question: str):
        docs = self.retriever.get_relevant_documents(question)
        print(f"[RAG] Retrieved {len(docs)} chunks")

        if not docs:
            return {
                "answer": "I do not know.",
                "confidence": 0.0,
                "source_documents": []
            }

        context = "\n\n".join(d.page_content for d in docs)

        prompt = f"""
You are an expert assistant for the Aave protocol.

Answer the question using ONLY the context below.
You may paraphrase or infer if the information is clearly implied.
Do NOT introduce unsupported facts.
If the context does not support an answer, say "I do not know".

Context:
{context}

Question:
{question}

Answer in 2–3 sentences.
"""

        response = self.llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)

        score = grounding_score(answer, docs)
        confidence = max(score, 0.35)

        return {
            "answer": answer,
            "confidence": confidence,
            "source_documents": docs
        }
