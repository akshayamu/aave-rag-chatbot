import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parents[1]))

import json
import time
import csv
import numpy as np
from src.rag_pipeline import AaveRAGPipeline

# ------------------------------
# Configuration
# ------------------------------
K = 5
rag = AaveRAGPipeline(k=K)

qa_path = "eval/qa.jsonl"
out_path = "eval/results.csv"

rows = []

# ------------------------------
# Evaluation loop
# ------------------------------
with open(qa_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        ex = json.loads(line)

        # ------------------------------
        # End-to-end latency (RAG query)
        # ------------------------------
        start = time.time()
        result = rag.query(ex["question"])
        latency_ms = (time.time() - start) * 1000

        answer = result["answer"]
        docs = result["source_documents"]

        # ------------------------------
        # Retrieval metrics (Recall@K, MRR)
        # ------------------------------
        sources = [d.metadata.get("source") for d in docs]

        hit = int(ex["source_doc"] in sources)
        recall_k = hit  # single ground-truth document

        rank = sources.index(ex["source_doc"]) + 1 if hit else None
        mrr = 1.0 / rank if rank else 0.0

        # ------------------------------
        # Faithfulness / Groundedness
        # ------------------------------
        grounded = int(
            any(
                ex["answer"].lower() in d.page_content.lower()
                for d in docs
            )
        )

        rows.append({
            "id": ex["id"],
            "recall@5": recall_k,
            "mrr": round(mrr, 4),
            "grounded": grounded,
            "latency_ms": round(latency_ms, 2),
        })

# ------------------------------
# Aggregate latency statistics
# ------------------------------
if not rows:
    raise ValueError("No evaluation rows produced. Check qa.jsonl formatting.")

latencies = [r["latency_ms"] for r in rows]

p50 = np.percentile(latencies, 50)
p95 = np.percentile(latencies, 95)

print(f"Latency p50: {p50:.2f} ms")
print(f"Latency p95: {p95:.2f} ms")

# ------------------------------
# Save results
# ------------------------------
Path("eval").mkdir(exist_ok=True)

with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("Evaluation saved to eval/results.csv")
