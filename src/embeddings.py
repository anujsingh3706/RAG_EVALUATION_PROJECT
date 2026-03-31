import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

class EmbeddingModel:
    """
    Wraps sentence-transformers for creating dense vector embeddings.
    Model: all-MiniLM-L6-v2 (fast, lightweight, 384-dim, great for RAG)
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        print(f"\n=== LOADING EMBEDDING MODEL: {self.MODEL_NAME} ===")
        self.model = SentenceTransformer(self.MODEL_NAME)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"  Embedding dimension: {self.dimension}")

    def embed(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """
        Embed a list of strings.
        Returns: numpy array of shape (len(texts), dimension)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True   # L2-normalize for cosine similarity via dot product
        )
        return embeddings

    def embed_single(self, text: str) -> np.ndarray:
        """Embed a single string. Returns shape (dimension,)"""
        return self.embed([text])[0]