# src/app.py
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from rag_pipeline import AaveRAGPipeline
from setup_env import setup_groq_api_key
from config import GROQ_API_KEY

# -------------------------
# Streamlit Page Config
# -------------------------
st.set_page_config(
    page_title="Aave RAG Chatbot v2 – Zero Trace",
    page_icon="💬",
    layout="wide"
)

# Custom CSS for better UI (fixed white text issue)
st.markdown("""
<style>
    .answer-container {
        background-color: #ffffff;   /* pure white background */
        color: #000000;              /* black text */
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        border: 1px solid #ddd;
        font-size: 16px;
        line-height: 1.6;
    }
    .confidence-meter {
        height: 10px;
        background-color: #e9ecef;
        border-radius: 5px;
        margin: 10px 0;
        overflow: hidden;
    }
    .confidence-level {
        height: 100%;
        background-color: #28a745;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Title
# -------------------------
st.title("💬 Aave RAG Chatbot v2 – Zero Trace")

# -------------------------
# API Key Setup
# -------------------------
if "api_key_set" not in st.session_state:
    st.session_state.api_key_set = bool(GROQ_API_KEY)

if not st.session_state.api_key_set:
    st.warning("⚠️ Groq API key not found. Please set it up first.")
    if st.button("Setup Groq API Key"):
        setup_groq_api_key()
        st.session_state.api_key_set = True
        st.experimental_rerun()
    st.stop()

# -------------------------
# Initialize Pipeline
# -------------------------
try:
    with st.spinner("🔄 Loading Aave RAG Pipeline..."):
        pipeline = AaveRAGPipeline()
except Exception as e:
    st.error(f"❌ Failed to load pipeline: {str(e)}")
    st.stop()

# -------------------------
# Main Input
# -------------------------
st.markdown("### ❓ Ask about Aave:")
user_question = st.text_input("", placeholder="e.g., How do I supply assets to Aave?")

# -------------------------
# Process Query
# -------------------------
if user_question:
    with st.spinner("🤔 Thinking..."):
        try:
            result = pipeline.ask(user_question)

            # Answer
            st.markdown("### ✅ Answer:")
            st.markdown(
                f"<div class='answer-container'>{result['answer']}</div>",
                unsafe_allow_html=True
            )

            # Confidence
            confidence_percentage = min(100, max(0, int(result.get("confidence", 0.75) * 100)))
            st.markdown("### 📊 Confidence:")
            st.markdown(f"**{confidence_percentage}%**")

            st.markdown(
                f"""
                <div class="confidence-meter">
                    <div class="confidence-level" style="width: {confidence_percentage}%"></div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Sources
            with st.expander("📂 Source Documents"):
                if result.get("source_documents"):
                    for i, doc in enumerate(result["source_documents"]):
                        st.markdown(f"**Document {i+1}:**")
                        st.text(doc.page_content[:1000])  # limit for performance
                        st.markdown("---")
                else:
                    st.info("No source documents available.")

        except Exception as e:
            st.error(f"⚠️ Error while processing your question: {str(e)}")

# -------------------------
# Sidebar Info
# -------------------------
with st.sidebar:
    st.markdown("## ℹ️ About")
    st.markdown("""
    This chatbot uses Retrieval-Augmented Generation (RAG) to answer questions about **Aave**.
    
    **Stack:**
    - Ollama (all-minilm) → Embeddings
    - Groq API (llama-3.1-8b-instant) → LLM
    - FAISS → Vector storage
    - LangChain → Orchestration
    """)

    st.markdown("## 🔎 How it works")
    st.markdown("""
    1. Your question → Embedded with Ollama  
    2. Retrieve relevant chunks from FAISS  
    3. Groq LLM → Generates an answer  
    4. Sources shown for transparency  
    """)

    st.markdown("## 🔒 Privacy")
    st.markdown("""
    - No external data storage  
    - API keys stay in `.env`  
    - Local FAISS index  
    """)
