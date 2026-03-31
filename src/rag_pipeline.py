import os
from src.ingestion import ingest_all_documents
from src.embeddings import EmbeddingModel
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.generator import Generator

VECTOR_STORE_PATH = "vector_store_cache"

class RAGPipeline:
    """
    Orchestrates the full RAG pipeline:
    Ingest → Embed → Store → Retrieve → Generate
    """

    def __init__(self, docs_dir: str = "data/documents", top_k: int = 4, force_rebuild: bool = False):
        self.docs_dir = docs_dir
        self.top_k = top_k

        # Initialize core components
        self.embedding_model = EmbeddingModel()
        self.vector_store = VectorStore(dimension=self.embedding_model.dimension)
        self.generator = Generator()

        # Build or load vector store
        if not force_rebuild and os.path.exists(VECTOR_STORE_PATH):
            print("\n=== LOADING CACHED VECTOR STORE ===")
            self.vector_store.load(VECTOR_STORE_PATH)
        else:
            self._build_index()

        self.retriever = Retriever(self.vector_store, self.embedding_model, top_k=self.top_k)

    def _build_index(self):
        """Ingest documents, embed chunks, and build the FAISS index."""
        print("\n=== BUILDING VECTOR INDEX FROM SCRATCH ===")

        # Step 1: Load & chunk documents
        chunks = ingest_all_documents(self.docs_dir)

        # Step 2: Embed all chunks
        print("\n=== EMBEDDING CHUNKS ===")
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.embed(texts, show_progress=True)

        # Step 3: Store in FAISS
        self.vector_store.add_chunks(chunks, embeddings)

        # Step 4: Persist to disk
        self.vector_store.save(VECTOR_STORE_PATH)
        print("  Index built and saved successfully.")

    def query(self, question: str, use_dedup: bool = True) -> dict:
        """
        Run the full RAG pipeline for a single question.
        Returns a structured result dict.
        """
        # Retrieve
        if use_dedup:
            retrieved = self.retriever.retrieve_with_dedup(question)
        else:
            retrieved = self.retriever.retrieve(question)

        # Format context
        context = self.retriever.format_context(retrieved)

        # Generate
        result = self.generator.generate_with_citation(question, retrieved, context)
        return result

    def query_batch(self, questions: list, use_dedup: bool = True) -> list:
        """Run the pipeline for a batch of questions."""
        results = []
        for i, question in enumerate(questions, 1):
            print(f"  Processing question {i}/{len(questions)}: {question[:60]}...")
            result = self.query(question, use_dedup=use_dedup)
            results.append(result)
        return results