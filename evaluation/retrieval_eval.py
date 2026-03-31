from typing import List, Tuple
from src.ingestion import DocumentChunk


def precision_at_k(
    retrieved: List[Tuple[DocumentChunk, float]],
    relevant_doc_id: str,
    k: int = 4
) -> float:
    """
    Precision@K: What fraction of top-K retrieved chunks
    come from the expected source document?
    """
    top_k = retrieved[:k]
    relevant_count = sum(
        1 for chunk, _ in top_k
        if chunk.doc_id == relevant_doc_id
    )
    return round(relevant_count / k, 4)


def recall_at_k(
    retrieved: List[Tuple[DocumentChunk, float]],
    relevant_doc_id: str,
    k: int = 4
) -> float:
    """
    Recall@K: Did the retriever find at least one chunk
    from the relevant document in the top-K results?
    (Binary: 1.0 if yes, 0.0 if no)
    """
    top_k = retrieved[:k]
    found = any(chunk.doc_id == relevant_doc_id for chunk, _ in top_k)
    return 1.0 if found else 0.0


def reciprocal_rank(
    retrieved: List[Tuple[DocumentChunk, float]],
    relevant_doc_id: str
) -> float:
    """
    Reciprocal Rank (RR): 1/rank of the first relevant chunk.
    Used to compute Mean Reciprocal Rank (MRR) across questions.
    """
    for rank, (chunk, _) in enumerate(retrieved, 1):
        if chunk.doc_id == relevant_doc_id:
            return round(1.0 / rank, 4)
    return 0.0


def average_precision(
    retrieved: List[Tuple[DocumentChunk, float]],
    relevant_doc_id: str
) -> float:
    """
    Average Precision (AP): Average of P@k values at each
    position where a relevant chunk is retrieved.
    """
    hits = 0
    precision_sum = 0.0
    for rank, (chunk, _) in enumerate(retrieved, 1):
        if chunk.doc_id == relevant_doc_id:
            hits += 1
            precision_sum += hits / rank
    if hits == 0:
        return 0.0
    return round(precision_sum / hits, 4)


def retrieval_score_summary(
    retrieved: List[Tuple[DocumentChunk, float]],
    relevant_doc_id: str,
    k: int = 4
) -> dict:
    """Compute all retrieval metrics for one question."""
    scores = [score for _, score in retrieved[:k]]
    return {
        "relevant_doc_id":  relevant_doc_id,
        "retrieved_docs":   [c.doc_id for c, _ in retrieved[:k]],
        "retrieval_scores": [round(s, 4) for s in scores],
        "precision_at_k":   precision_at_k(retrieved, relevant_doc_id, k),
        "recall_at_k":      recall_at_k(retrieved, relevant_doc_id, k),
        "reciprocal_rank":  reciprocal_rank(retrieved, relevant_doc_id),
        "average_precision": average_precision(retrieved, relevant_doc_id),
        "hit": recall_at_k(retrieved, relevant_doc_id, k) == 1.0
    }