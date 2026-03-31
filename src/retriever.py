from typing import List, Tuple
from src.ingestion import DocumentChunk
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore


class Retriever:
    """
    Retrieves the most relevant document chunks for a given query.
    Supports both standard top-k retrieval and MMR-style deduplication.
    """

    def __init__(self, vector_store: VectorStore, embedding_model: EmbeddingModel, top_k: int = 4):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int = None) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieve top-k chunks most relevant to the query.
        Returns: list of (DocumentChunk, similarity_score) sorted by score desc
        """
        k = top_k or self.top_k
        query_embedding = self.embedding_model.embed_single(query)
        results = self.vector_store.search(query_embedding, top_k=k)
        return results

    def retrieve_with_dedup(self, query: str, top_k: int = None) -> List[Tuple[DocumentChunk, float]]:
        """
        Retrieve top-k chunks, deduplicating by source document.
        Ensures we don't return two chunks from the exact same doc if avoidable.
        """
        k = top_k or self.top_k
        # Fetch more candidates, then filter
        candidates = self.retrieve(query, top_k=k * 3)
        seen_docs = set()
        filtered = []
        for chunk, score in candidates:
            if chunk.doc_id not in seen_docs:
                filtered.append((chunk, score))
                seen_docs.add(chunk.doc_id)
            if len(filtered) >= k:
                break
        # If we don't have enough unique docs, fill remaining from candidates
        if len(filtered) < k:
            for chunk, score in candidates:
                if (chunk, score) not in filtered:
                    filtered.append((chunk, score))
                if len(filtered) >= k:
                    break
        return filtered

    def format_context(self, retrieved: List[Tuple[DocumentChunk, float]]) -> str:
        """Format retrieved chunks into a single context string for the LLM prompt."""
        context_parts = []
        for i, (chunk, score) in enumerate(retrieved, 1):
            context_parts.append(
                f"[Source {i}: {chunk.metadata['title']} | Score: {score:.3f}]\n{chunk.content}"
            )
        return "\n\n---\n\n".join(context_parts)