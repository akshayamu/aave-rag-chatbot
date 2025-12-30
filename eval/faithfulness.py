def groundedness_score(answer, retrieved_chunks):
    """
    Measures how much of the answer is supported by retrieved context.
    """
    supported = 0
    sentences = [s.strip() for s in answer.split(".") if s.strip()]

    for sentence in sentences:
        if any(sentence.lower() in chunk.lower() for chunk in retrieved_chunks):
            supported += 1

    return supported / max(len(sentences), 1)
