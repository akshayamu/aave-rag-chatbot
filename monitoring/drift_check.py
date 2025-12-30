import json
import csv
import numpy as np

BASELINE_PATH = "monitoring/baseline.json"
RESULTS_PATH = "eval/results.csv"

# ------------------------------
# Load baseline
# ------------------------------
with open(BASELINE_PATH, "r") as f:
    baseline = json.load(f)

# ------------------------------
# Load latest eval results
# ------------------------------
rows = []
with open(RESULTS_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

if not rows:
    raise RuntimeError("No evaluation results found.")

recall_vals = [int(r["recall@5"]) for r in rows]
mrr_vals = [float(r["mrr"]) for r in rows]
grounded_vals = [int(r["grounded"]) for r in rows]
latencies = [float(r["latency_ms"]) for r in rows]

current_metrics = {
    "recall@5_mean": np.mean(recall_vals),
    "mrr_mean": np.mean(mrr_vals),
    "grounded_rate": np.mean(grounded_vals),
    "latency_p95_ms": np.percentile(latencies, 95),
}

# ------------------------------
# Drift checks
# ------------------------------
alerts = []

if current_metrics["recall@5_mean"] < baseline["recall@5_mean"] - 0.2:
    alerts.append("Recall@5 degraded significantly")

if current_metrics["mrr_mean"] < baseline["mrr_mean"] - 0.1:
    alerts.append("MRR degraded significantly")

if current_metrics["latency_p95_ms"] > baseline["latency_p95_ms"] * 1.5:
    alerts.append("Latency p95 increased sharply")

# ------------------------------
# Output
# ------------------------------
print("Current metrics:", current_metrics)

if alerts:
    print("⚠️ DRIFT ALERTS:")
    for a in alerts:
        print("-", a)
else:
    print("✅ System healthy (no significant drift detected)")
