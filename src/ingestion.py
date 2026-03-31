import os
from dataclasses import dataclass
from typing import List

@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    content: str
    metadata: dict

def load_documents(docs_dir: str) -> List[dict]:
    """Load all .txt files from the documents directory."""
    documents = []
    for filename in sorted(os.listdir(docs_dir)):
        if filename.endswith(".txt"):
            filepath = os.path.join(docs_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read().strip()
            doc_id = filename.replace(".txt", "")
            # Extract title from first line
            lines = content.split("\n")
            title = lines[0].replace("Title: ", "").strip() if lines[0].startswith("Title:") else doc_id
            documents.append({
                "doc_id": doc_id,
                "title": title,
                "content": content,
                "filepath": filepath
            })
            print(f"  [LOADED] {filename} ({len(content)} chars)")
    return documents


def chunk_document(doc: dict, chunk_size: int = 400, overlap: int = 80) -> List[DocumentChunk]:
    """
    Split a document into overlapping word-based chunks.
    chunk_size: number of words per chunk
    overlap: number of words to overlap between consecutive chunks
    """
    words = doc["content"].split()
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        chunk_id = f"{doc['doc_id']}_chunk{chunk_index}"

        chunks.append(DocumentChunk(
            chunk_id=chunk_id,
            doc_id=doc["doc_id"],
            content=chunk_text,
            metadata={
                "title": doc["title"],
                "doc_id": doc["doc_id"],
                "chunk_index": chunk_index,
                "word_start": start,
                "word_end": end
            }
        ))
        chunk_index += 1
        if end == len(words):
            break
        start += chunk_size - overlap  # move forward with overlap

    return chunks


def ingest_all_documents(docs_dir: str, chunk_size: int = 400, overlap: int = 80) -> List[DocumentChunk]:
    """Full ingestion pipeline: load → chunk all documents."""
    print("\n=== DOCUMENT INGESTION ===")
    documents = load_documents(docs_dir)
    all_chunks = []
    for doc in documents:
        chunks = chunk_document(doc, chunk_size, overlap)
        all_chunks.extend(chunks)
        print(f"  [CHUNKED] {doc['doc_id']} → {len(chunks)} chunks")
    print(f"\n  Total chunks created: {len(all_chunks)}")
    return all_chunks