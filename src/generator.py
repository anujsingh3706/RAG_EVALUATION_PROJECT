import os
from groq import Groq
from dotenv import load_dotenv
from typing import List, Tuple
from src.ingestion import DocumentChunk

load_dotenv()

SYSTEM_PROMPT = """You are a precise and factual assistant specializing in the history and science of quantum computing.

You will be given a QUESTION and CONTEXT extracted from a curated set of documents.

Your task:
1. Answer the question ONLY using information from the provided context.
2. Be concise and factually accurate. Do not add information not present in the context.
3. If the context does not contain enough information to answer, say: "The provided documents do not contain sufficient information to answer this question."
4. Do not mention "the context" or "the documents" in your answer — answer naturally as an expert.
"""

class Generator:
    """
    Uses Groq's LLaMA 3.3 70B to generate answers grounded in retrieved context.
    """

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        print(f"\n=== GENERATOR INITIALIZED (model={self.model}) ===")

    def generate(self, question: str, context: str, max_tokens: int = 512) -> str:
        """
        Generate an answer to the question using the provided context.
        """
        user_message = f"""CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            max_tokens=max_tokens,
            temperature=0.1   # Low temperature for factual, deterministic answers
        )
        return response.choices[0].message.content.strip()

    def generate_with_citation(self, question: str,
                                retrieved: List[Tuple[DocumentChunk, float]],
                                context: str) -> dict:
        """
        Generate answer and return full structured result with sources.
        """
        answer = self.generate(question, context)
        sources = [
            {
                "doc_id": chunk.doc_id,
                "title": chunk.metadata["title"],
                "score": round(score, 4),
                "chunk_id": chunk.chunk_id
            }
            for chunk, score in retrieved
        ]
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "context_used": context
        }