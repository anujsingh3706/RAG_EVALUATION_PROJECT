import os
import json
import pickle
import numpy as np
import faiss
from typing import List, Tuple
from src.ingestion import DocumentChunk

class VectorStore:
    """
    FAISS-backed vector store for storing and retrieving document chunk embeddings.
    Uses Inner Product search (equivalent to cosine similarity on normalized vectors).
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product = cosine on normalized vecs
        self.chunks: List[DocumentChunk] = []      # Parallel list to FAISS index rows
        print(f"\n=== VECTOR STORE INITIALIZED (dim={dimension}) ===")

    def add_chunks(self, chunks: List[DocumentChunk], embeddings: np.ndarray):
        """Add chunks and their embeddings to the store."""
        assert len(chunks) == len(embeddings), "Chunks and embeddings must be same length"
        embeddings = embeddings.astype(np.float32)
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        print(f"  Added {len(chunks)} chunks to vector store (total: {self.index.ntotal})")

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for top-k most similar chunks to the query embedding.
        Returns: list of (DocumentChunk, similarity_score) tuples
        """
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:   # FAISS returns -1 for unfilled slots
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, path: str):
        """Persist the FAISS index and chunk metadata to disk."""
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "chunks.pkl"), "wb") as f:
            pickle.dump(self.chunks, f)
        print(f"  Vector store saved to: {path}")

    def load(self, path: str):
        """Load a previously saved vector store from disk."""
        self.index = faiss.read_index(os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "chunks.pkl"), "rb") as f:
            self.chunks = pickle.load(f)
        print(f"  Vector store loaded from: {path} ({self.index.ntotal} vectors)")