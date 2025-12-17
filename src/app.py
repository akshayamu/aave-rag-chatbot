import streamlit as st
from rag_pipeline import AaveRAGPipeline


st.set_page_config(page_title="Aave RAG Chatbot", layout="centered")

st.title("Aave RAG Chatbot")
st.caption("Answers are grounded in official Aave documentation and include source citations.")

rag = AaveRAGPipeline()

example_questions = [
    "What triggers liquidation in Aave?",
    "What is the health factor in Aave and why is it important?",
    "How does partial liquidation work in Aave?",
    "What is the difference between stable and variable borrow rates in Aave?",
    "How does governance work in the Aave protocol?"
]

selected = st.selectbox("Try an example question:", example_questions)

question = st.text_input(
    "Or ask your own question:",
    value=selected
)

if st.button("Ask"):
    with st.spinner("Searching Aave documentation..."):
        try:
            result = rag.query(question)

            st.markdown("### Answer")
            st.write(result["answer"])

            st.markdown(f"**Confidence:** {result['confidence']:.2f}")

            if result["confidence"] < 0.3:
                st.warning("⚠️ Low confidence. Relevant documentation may not have been retrieved.")

            st.markdown("### Sources")
            if not result["source_documents"]:
                st.info("No relevant sources retrieved.")
            else:
                for doc in result["source_documents"]:
                    st.write(doc.metadata.get("source"))

        except Exception as e:
            st.error(f"Error while processing your question: {e}")
