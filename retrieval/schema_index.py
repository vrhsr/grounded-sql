"""
retrieval/schema_index.py — FAISS semantic index over schema descriptions

Encodes schema+question pairs as dense vectors using sentence-transformers.
Used for semantic retrieval in the RAG and fine-tuned+RAG systems.

Usage:
    python retrieval/schema_index.py --build --train data/processed/train.jsonl
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
from pathlib import Path

from tqdm import tqdm
from rich.console import Console

console = Console()

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    RETRIEVAL_AVAILABLE = True
except ImportError:
    RETRIEVAL_AVAILABLE = False
    console.print("[yellow]Warning: faiss-cpu or sentence-transformers not installed. Retrieval disabled.[/yellow]")


class FAISSSchemaIndex:
    """
    Dense semantic index for schema+question pairs.
    Uses 'all-MiniLM-L6-v2' for fast, good-quality embeddings.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        if not RETRIEVAL_AVAILABLE:
            raise ImportError("Install faiss-cpu and sentence-transformers")
        self.encoder = SentenceTransformer(self.MODEL_NAME)
        self.index = None
        self.metadata = []  # list of dicts: {question, gold_sql, db_id, schema_snippet}

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self.encoder.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,  # cosine similarity via dot product
        ).astype(np.float32)

    def build(self, train_samples: list[dict]) -> None:
        """Build FAISS index from training samples."""
        console.print(f"\n[bold]Building FAISS index over {len(train_samples)} samples...[/bold]")

        texts = []
        for s in train_samples:
            # Combine question + schema snippet as the retrieval text
            schema_snippet = s.get("prompt", "")[-200:]  # last 200 chars of prompt
            query_text = f"{s.get('question', '')} {schema_snippet}"
            texts.append(query_text)
            self.metadata.append({
                "question": s.get("question", ""),
                "gold_sql": s.get("gold_sql", ""),
                "db_id": s.get("db_id", ""),
                "schema_snippet": schema_snippet,
            })

        embeddings = self._encode(texts)
        dim = embeddings.shape[1]

        # IVF index for fast approximate search
        n_clusters = min(int(np.sqrt(len(texts))), 256)
        quantizer = faiss.IndexFlatIP(dim)
        self.index = faiss.IndexIVFFlat(quantizer, dim, n_clusters, faiss.METRIC_INNER_PRODUCT)
        self.index.train(embeddings)
        self.index.add(embeddings)
        self.index.nprobe = 32  # search 32 clusters at query time

        console.print(f"[green]✓[/green] Index built: {self.index.ntotal} vectors, dim={dim}")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve top_k most similar examples."""
        if self.index is None:
            return []
        query_emb = self._encode([query])
        scores, indices = self.index.search(query_emb, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                r = dict(self.metadata[idx])
                r["score"] = float(score)
                results.append(r)
        return results

    def save(self, dir_path: str) -> None:
        os.makedirs(dir_path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(dir_path, "schema.faiss"))
        with open(os.path.join(dir_path, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)
        console.print(f"[green]✓[/green] Index saved to {dir_path}/")

    @classmethod
    def load(cls, dir_path: str) -> "FAISSSchemaIndex":
        instance = cls()
        instance.index = faiss.read_index(os.path.join(dir_path, "schema.faiss"))
        with open(os.path.join(dir_path, "metadata.pkl"), "rb") as f:
            instance.metadata = pickle.load(f)
        console.print(f"[green]✓[/green] Loaded FAISS index: {instance.index.ntotal} vectors")
        return instance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--train", default="data/processed/train.jsonl")
    parser.add_argument("--index-dir", default="retrieval/indexes/faiss")
    parser.add_argument("--query", default=None, help="Test query to retrieve examples for")
    args = parser.parse_args()

    if args.build:
        with open(args.train) as f:
            samples = [json.loads(l) for l in f if l.strip()]
        idx = FAISSSchemaIndex()
        idx.build(samples)
        idx.save(args.index_dir)

    if args.query:
        idx = FAISSSchemaIndex.load(args.index_dir)
        results = idx.search(args.query, top_k=3)
        for i, r in enumerate(results):
            console.print(f"\n[{i+1}] Score: {r['score']:.3f}")
            console.print(f"  Q: {r['question']}")
            console.print(f"  SQL: {r['gold_sql'][:80]}")
