# src/rag_pipeline.py
import os
import re
from dotenv import load_dotenv

# ensure env variables are loaded (so GROQ_API_KEY is available)
load_dotenv()

# allow running from src/ directory
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# use your existing loader so vectorstore format matches what you built
from document_processor import load_vectorstore
from config import GROQ_API_KEY, GROQ_LLM_MODEL, OLLAMA_EMBEDDING_MODEL


def _highlight_answer(answer: str, source_docs):
    """Find a sentence from source docs that best matches the answer and bold it."""
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
    def __init__(self, model_path="models/aave_faiss_index", k=3):
        """
        Minimal constructor: loads persisted vectorstore and instantiates Groq LLM and QA chain.
        Assumes you already built and saved the vectorstore with build_vectorstore.py.
        """
        self.model_path = model_path
        self.k = k

        # Sanity checks
        if not os.path.exists(self.model_path) and not os.path.exists(f"{self.model_path}.faiss"):
            raise FileNotFoundError(
                f"Vectorstore not found at '{self.model_path}'. Run 'python src/build_vectorstore.py' first."
            )

        # Load vectorstore using your document_processor load_vectorstore (keeps consistency)
        # Pass model_name if your load_vectorstore expects it; otherwise it should use default.
        try:
            # try to pass model name if loader supports it
            self.vectorstore = load_vectorstore(path=self.model_path, model_name=OLLAMA_EMBEDDING_MODEL)
        except TypeError:
            # fallback: loader might accept only path
            self.vectorstore = load_vectorstore(self.model_path)

        # Create retriever
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k})

        # Validate API key
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY missing. Please set it in your .env or via setup_groq_api_key().")

        # Instantiate Groq LLM (handle different param names across versions)
        try:
            self.llm = ChatGroq(groq_api_key=GROQ_API_KEY, model=GROQ_LLM_MODEL, temperature=0.0)
        except TypeError:
            # fallback if ChatGroq expects model_name
            self.llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name=GROQ_LLM_MODEL, temperature=0.0)

        # Prompt template (keeps it clear and deterministic)
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

        # Build RetrievalQA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type="stuff",
            chain_type_kwargs={"prompt": self.prompt},
            return_source_documents=True,
        )

    def ask(self, question: str):
        """
        Ask a question and return a dict:
          { "answer": str, "confidence": float (0..1), "source_documents": [docs...] }
        """
        if not question or not question.strip():
            return {"answer": "⚠️ Please enter a valid question.", "confidence": 0.0, "source_documents": []}

        result = self.qa_chain({"query": question})

        answer = result.get("result", "")
        source_docs = result.get("source_documents", []) or []

        # Try to extract similarity / score metadata if available
        scores = []
        for d in source_docs:
            md = getattr(d, "metadata", {}) or {}
            # common keys used by some retrievers
            for key in ("score", "similarity", "cosine_score", "similarity_score", "query_score"):
                if key in md and md[key] is not None:
                    try:
                        scores.append(float(md[key]))
                    except Exception:
                        pass

        if scores:
            # assume scores are similarity in 0..1; normalize if clearly >1
            confidence = max(scores)
            if confidence > 1 and confidence <= 100:
                confidence = confidence / 100.0
            confidence = max(0.0, min(1.0, confidence))
        else:
            # fallback: simple heuristic used previously (keeps behavior you liked)
            confidence = 0.8 if source_docs else 0.4

        # Also compute highlight match score and bump confidence if necessary
        highlighted_answer, match_score = _highlight_answer(answer, source_docs)
        # match_score is 0..1; combine conservatively
        confidence = max(confidence, match_score)

        return {"answer": highlighted_answer, "confidence": float(confidence), "source_documents": source_docs}

    # alias for backward compatibility
    def query(self, question: str):
        return self.ask(question)
