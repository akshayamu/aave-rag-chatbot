import streamlit as st
from rag_pipeline import AaveRAGPipeline

# Initialize pipeline
if "pipeline" not in st.session_state:
    st.session_state.pipeline = AaveRAGPipeline()

st.title("🤖 Aave RAG Chatbot")

if st.button("Load Knowledge Base"):
    try:
        st.session_state.pipeline.load_and_index()
        st.success("✅ Knowledge base loaded successfully!")
    except Exception as e:
        st.error(f"Error: {e}")

query = st.text_input("Ask a question about Aave:")

if query:
    try:
        answer, sources = st.session_state.pipeline.ask(query)
        st.markdown(f"### 📝 Answer:\n{answer}")
        st.markdown("### 📂 Sources:")
        for s in sources:
            st.markdown(f"- {s}")
    except Exception as e:
        st.error(f"⚠️ Error while processing your question: {e}")
