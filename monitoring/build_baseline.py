import csv
import json
import numpy as np
from pathlib import Path

RESULTS_PATH = "eval/results.csv"
BASELINE_PATH = "monitoring/baseline.json"

rows = []
with open(RESULTS_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

if not rows:
    raise RuntimeError("No rows found in results.csv")

recall_vals = [int(r["recall@5"]) for r in rows]
mrr_vals = [float(r["mrr"]) for r in rows]
grounded_vals = [int(r["grounded"]) for r in rows]
latencies = [float(r["latency_ms"]) for r in rows]

baseline = {
    "recall@5_mean": round(np.mean(recall_vals), 4),
    "mrr_mean": round(np.mean(mrr_vals), 4),
    "grounded_rate": round(np.mean(grounded_vals), 4),
    "latency_p95_ms": round(np.percentile(latencies, 95), 2),
}

Path("monitoring").mkdir(exist_ok=True)

with open(BASELINE_PATH, "w", encoding="utf-8") as f:
    json.dump(baseline, f, indent=2)

print("Baseline written to monitoring/baseline.json")
print(baseline)
