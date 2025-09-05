import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from setup_env import setup_groq_api_key
from config import GROQ_API_KEY
from rag_pipeline import AaveRAGPipeline

st.set_page_config(page_title="Aave RAG Chatbot v2 – Zero Trace", page_icon="💬", layout="wide")

st.markdown(
    """
    <style>
    .answer-container {
        background-color: #ffffff;
        color: #000000;
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
    """,
    unsafe_allow_html=True,
)

st.title("💬 Aave RAG Chatbot v2 – Zero Trace")

if "api_key_set" not in st.session_state:
    st.session_state.api_key_set = bool(GROQ_API_KEY)

if not st.session_state.api_key_set:
    st.warning("⚠️ Groq API key not found. Please set it up first.")
    if st.button("Setup Groq API Key"):
        setup_groq_api_key()
        st.session_state.api_key_set = True
        st.experimental_rerun()
    st.stop()

@st.cache_resource(show_spinner=True)
def load_pipeline():
    return AaveRAGPipeline()

try:
    with st.spinner("🔄 Loading Aave RAG Pipeline..."):
        pipeline = load_pipeline()
except Exception as e:
    st.error(f"❌ Failed to load pipeline: {str(e)}")
    st.stop()

st.markdown("### ❓ Ask about Aave:")

user_question = st.text_input("", placeholder="e.g., How do I supply assets to Aave?")
if user_question:
    with st.spinner("🤔 Thinking..."):
        try:
            result = pipeline.ask(user_question)
        except Exception as e:
            st.error(f"⚠️ Error while processing your question: {str(e)}")
            result = None

    if result:
        st.markdown("### ✅ Answer:")
        st.markdown(f"<div class='answer-container'>{result['answer']}</div>", unsafe_allow_html=True)

        raw_conf = result.get("confidence", 0.0)
        try:
            conf_pct = int(raw_conf * 100) if raw_conf <= 1.0 else int(raw_conf)
        except Exception:
            conf_pct = 0

        st.markdown("### 📊 Confidence:")

        try:
            st.progress(conf_pct / 100)
        except Exception:
            pass
        st.markdown(f"**{conf_pct}%**")

        with st.expander("📂 Source Documents"):
            if result.get("source_documents"):
                for i, doc in enumerate(result["source_documents"], start=1):
                    st.markdown(f"**Document {i}:**")
                    st.text(getattr(doc, "page_content", "")[:1000])
                    st.markdown("---")
            else:
                st.info("No source documents available.")
