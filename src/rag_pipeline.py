import os
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

try:
    from document_processor import load_vectorstore
except Exception:
    load_vectorstore = None

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config import GROQ_API_KEY, GROQ_LLM_MODEL, OLLAMA_EMBEDDING_MODEL

def _highlight_answer(answer: str, source_docs):
    full_text = " ".join([getattr(d, "page_content", "") for d in source_docs])
    sentences = re.split(r"[.!?]+\s*", full_text)
    answer_words = set(answer.lower().split())
    best_match = ""
    best_score = 0.0
    for s in sentences:
        s = s.strip()
        if len(s) < 10:
            continue
        s_words = set(s.lower().split())
        if not s_words:
            continue
        inter = len(answer_words.intersection(s_words))
        union = len(answer_words.union(s_words))
        score = (inter / union) if union > 0 else 0.0
        if score > best_score:
            best_score = score
            best_match = s
    if best_score > 0.05 and best_match:
        try:
            pattern = re.escape(best_match.strip())
            highlighted = re.sub(pattern, f"**{best_match.strip()}**", answer, flags=re.IGNORECASE)
            return highlighted, best_score
        except Exception:
            return answer, best_score
    return answer, best_score

class AaveRAGPipeline:
    def __init__(self, model_path="models/aave_faiss_index", k=3, data_folder="data"):
        self.model_path = model_path
        self.k = k
        self.data_folder = Path(data_folder)

        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY missing. Set it in .env or via setup_groq_api_key().")

        try:
            # ChatGroq may expect only model param
            self.llm = ChatGroq(model=GROQ_LLM_MODEL, temperature=0.0)
        except TypeError:
            # fallback older param name
            self.llm = ChatGroq(groq_api_key=GROQ_API_KEY, model=GROQ_LLM_MODEL, temperature=0.0)

        self.vectorstore = self._create_or_load_index()

        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k})
        self.prompt = PromptTemplate(
            template=(
                "You are an assistant for answering questions about Aave Protocol.\n"
                "Use the provided context to give accurate answers. If the context does not contain the answer, say you don't know.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            ),
            input_variables=["context", "question"],
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=True,
        )

    def _create_or_load_index(self):
        # Use project loader if any
        if load_vectorstore:
            try:
                return load_vectorstore(path=self.model_path, model_name=OLLAMA_EMBEDDING_MODEL)
            except TypeError:
                try:
                    return load_vectorstore(self.model_path)
                except Exception:
                    pass

        idx_dir = Path(self.model_path)
        if idx_dir.exists() or Path(f"{self.model_path}.faiss").exists():
            try:
                embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                return FAISS.load_local(str(idx_dir), embed, allow_dangerous_deserialization=True)
            except Exception:
                pass

        if not self.data_folder.exists():
            raise FileNotFoundError(f"Vectorstore '{self.model_path}' not found and data folder '{self.data_folder}' missing.")

        pdfs = [p for p in self.data_folder.iterdir() if p.suffix.lower() == ".pdf"]
        if not pdfs:
            raise FileNotFoundError(f"No PDFs in '{self.data_folder}' to build vectorstore.")

        docs = []
        for f in pdfs:
            docs.extend(PyPDFLoader(str(f)).load())

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)
        embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vs = FAISS.from_documents(splits, embed)
        Path(self.model_path).mkdir(parents=True, exist_ok=True)
        vs.save_local(str(self.model_path))
        return vs

    def ask(self, question: str):
        if not question or not question.strip():
            return {"answer": "⚠️ Please enter a valid question.", "confidence": 0.0, "source_documents": []}

        result = self.qa_chain({"query": question})
        answer = result.get("result", "")
        source_docs = result.get("source_documents", []) or []

        scores = []
        for d in source_docs:
            md = getattr(d, "metadata", {}) or {}
            for key in ("score", "similarity", "cosine_score", "similarity_score", "query_score"):
                if key in md and md[key] is not None:
                    try:
                        scores.append(float(md[key]))
                    except Exception:
                        pass

        confidence = max(scores) if scores else (0.8 if source_docs else 0.4)
        if confidence > 1 and confidence <= 100:
            confidence = confidence / 100.0
        confidence = max(0.0, min(1.0, float(confidence)))

        highlighted_answer, match_score = _highlight_answer(answer, source_docs)
        confidence = max(confidence, float(match_score))

        return {"answer": highlighted_answer, "confidence": confidence, "source_documents": source_docs}

    def query(self, question: str):
        return self.ask(question)
