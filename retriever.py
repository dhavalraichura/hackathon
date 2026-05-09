# ─────────────────────────────────────────────
# retriever.py — RAG retrieval engine
# Handles: embedding query, FAISS search, reranking
# ─────────────────────────────────────────────

import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder

import config


class Retriever:
    """
    Retrieval engine:
    - Embeds query using sentence-transformers
    - Searches FAISS index for top-k chunks
    - Optionally reranks with cross-encoder
    """

    def __init__(self, use_reranker: bool = False):
        print("🔧 Loading retriever...")

        # Embedding model (open-source, runs locally)
        self.embedder = SentenceTransformer(config.EMBEDDING_MODEL)

        # FAISS index
        index_file = os.path.join(config.FAISS_INDEX_PATH, "index.bin")
        if not os.path.exists(index_file):
            raise FileNotFoundError(
                f"FAISS index not found at {index_file}. "
                "Please run: python ingest.py first."
            )
        self.index = faiss.read_index(index_file)

        # Metadata (chunk text + source)
        with open(config.METADATA_PATH, "r") as f:
            self.metadata = json.load(f)

        # Optional reranker (cross-encoder — better precision, slower)
        self.use_reranker = use_reranker
        self.reranker = None
        if use_reranker:
            print("  ⚖  Loading reranker (cross-encoder)...")
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        print(f"  ✓ Retriever ready ({self.index.ntotal} vectors indexed)")


    def retrieve(self, query: str, top_k: int = config.TOP_K) -> list[dict]:
        """
        Retrieve top-k most relevant chunks for a query.
        Returns list of dicts with: text, source, score
        """
        # 1. Embed query
        query_vec = self.embedder.encode([query]).astype("float32")
        faiss.normalize_L2(query_vec)

        # 2. FAISS search — retrieve more than top_k if reranking
        fetch_k = top_k * 3 if self.use_reranker else top_k
        fetch_k = min(fetch_k, self.index.ntotal)

        scores, indices = self.index.search(query_vec, fetch_k)
        scores   = scores[0].tolist()
        indices  = indices[0].tolist()

        # 3. Build results
        results = []
        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self.metadata):
                continue
            chunk = self.metadata[idx]
            results.append({
                "text":   chunk["text"],
                "source": chunk["source"],
                "score":  round(float(score), 4),
                "id":     chunk["id"],
            })

        # 4. Reranking (optional bonus)
        if self.use_reranker and self.reranker and results:
            pairs   = [(query, r["text"]) for r in results]
            rescores = self.reranker.predict(pairs)
            for r, s in zip(results, rescores):
                r["rerank_score"] = round(float(s), 4)
            results.sort(key=lambda x: x["rerank_score"], reverse=True)

        return results[:top_k]


    def is_confident(self, results: list[dict]) -> bool:
        """Check if top result score is above confidence threshold."""
        if not results:
            return False
        top_score = results[0].get("rerank_score", results[0]["score"])
        return top_score >= config.CONFIDENCE_THRESHOLD
