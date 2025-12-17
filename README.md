# Aave RAG Chatbot — Evaluation-Driven Retrieval-Augmented Generation

An **evaluation-driven Retrieval-Augmented Generation (RAG) chatbot** built on
official **Aave protocol documentation**. The system emphasizes **grounded,
citation-backed answers**, explicit **uncertainty handling**, and **measured
latency**, rather than prompt engineering.

This project is designed as a **portfolio-grade RAG system**, demonstrating
best practices in retrieval evaluation, faithfulness checks, and transparent
system behavior.

---

## What This Project Does

- Answers questions about the Aave protocol using official documentation
- Grounds every response in retrieved source documents
- Displays citations and retrieved context
- Refuses to hallucinate when documentation does not support an answer
- Measures retrieval quality and end-to-end latency

---

## Data Sources

The knowledge base is built exclusively from official Aave documentation:

- Aave Protocol Overview (PDF)
- Aave V2 Technical Documentation
- Aave Governance Documentation

Only documentation content is indexed; UI artifacts and irrelevant text are
filtered during ingestion.

---

## Architecture

Aave Docs  
↓  
Document Chunking  
↓  
Sentence-Transformer Embeddings  
↓  
FAISS Vector Store  
↓  
Retriever (Top-K)  
↓  
Groq LLM (LLaMA-3.1-8B)  
↓  
Answer + Confidence + Citations  

---

## Evaluation

This project includes a **formal evaluation pipeline**, not just a demo UI.

### Evaluation Setup

- Vector Store: FAISS  
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`  
- LLM: Groq (LLaMA-3.1-8B via ChatGroq)  
- Evaluation Set: Curated QA pairs derived from Aave documentation  

### Metrics Tracked

- **Retrieval hit-rate@k**
- **Precision@k / Mean Reciprocal Rank (MRR)**
- **Faithfulness / grounding correctness**
- **End-to-end latency** (retrieval + generation)

### Results (Representative)

| Metric | Value |
|------|------|
| Latency p50 | ~338 ms |
| Latency p95 | ~5.3 s |
| Vector store | FAISS |
| Embeddings | MiniLM-L6-v2 |
| LLM | LLaMA-3.1-8B (Groq) |

Latency is measured end-to-end. Higher p95 reflects LLM inference variability
and cold starts, which are reported transparently.

---

## Failure-Aware Behavior

If retrieved documentation does not explicitly support an answer:

- The system returns **“I do not know”**
- Confidence is reduced
- Citations are still shown
- Hallucinations are avoided

This conservative behavior is intentional and evaluated.

---

## Streamlit Demo

The UI:
- Displays retrieved sources and context
- Shows confidence scores tied to evidence strength
- Makes uncertainty explicit when answers are unsupported

Example questions:
- What is the health factor in Aave?
- How is the health factor calculated?
- What happens when the health factor drops?
- What is the role of liquidators in Aave?

---

## How to Run Locally

### 1. Install dependencies
```bash
pip install -r requirements.txt
