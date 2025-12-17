## Evaluation

The Aave RAG chatbot is evaluated using a curated QA set derived from official
Aave protocol documentation. The evaluation focuses on retrieval quality,
answer grounding, and end-to-end latency.

### Metrics
- Retrieval hit-rate@k
- Precision@k / Mean Reciprocal Rank (MRR)
- Citation faithfulness (answer supported by retrieved context)
- End-to-end latency (retrieval + generation)

### Results

| Metric | Value |
|------|------|
| Latency p50 | 338 ms |
| Latency p95 | 5.33 s |
| Evaluation set size | 30 QA pairs |
| Vector store | FAISS |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| LLM | Groq (ChatGroq) |

Latency is measured end-to-end and includes retrieval and generation time.
Tail latency reflects LLM inference variability and cold starts.
