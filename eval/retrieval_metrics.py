import numpy as np

def recall_at_k(relevant_docs, retrieved_docs, k):
    retrieved_k = retrieved_docs[:k]
    return len(set(relevant_docs) & set(retrieved_k)) / max(len(relevant_docs), 1)

def reciprocal_rank(relevant_docs, retrieved_docs):
    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc in relevant_docs:
            return 1.0 / rank
    return 0.0

def mean_reciprocal_rank(all_relevant, all_retrieved):
    scores = [
        reciprocal_rank(rel, ret)
        for rel, ret in zip(all_relevant, all_retrieved)
    ]
    return float(np.mean(scores))
