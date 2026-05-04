"""
retrieval/hybrid_linker.py — Hybrid Schema Linker (BM25 + FAISS + RRF Fusion)

Combines sparse (BM25) and dense (FAISS) retrieval using Reciprocal Rank Fusion.

BM25: Great for exact column name matches ("customer_id", "created_at")
FAISS: Great for semantic paraphrases ("signed up date" → created_at)
RRF: Merges ranked lists without needing score calibration

This directly attacks hallucinated columns — the #1 failure mode.

Usage:
    python retrieval/hybrid_linker.py \
        --build \
        --tables dataset/spider/tables.json \
        --train data/processed/train.jsonl
    
    python retrieval/hybrid_linker.py \
        --query "total revenue by category for premium customers"
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Optional

from rich.console import Console

sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()

RRF_K = 60  # Standard RRF constant (from Cormack et al., 2009)


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict]],
    key: str = "db_id",
    score_key: str = "score",
    k: int = RRF_K,
) -> list[dict]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.
    
    RRF score = sum(1 / (k + rank_i)) for each list i
    
    This is order-invariant to score scales — no calibration needed
    between BM25 scores and cosine similarity.
    """
    rrf_scores = {}
    item_map = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            item_key = f"{item.get('db_id', '')}_{item.get('table_name', '')}_{item.get('question', '')}"
            rrf_scores[item_key] = rrf_scores.get(item_key, 0) + 1 / (k + rank)
            item_map[item_key] = item

    # Sort by RRF score descending
    sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
    results = []
    for key in sorted_keys:
        item = dict(item_map[key])
        item["rrf_score"] = rrf_scores[key]
        results.append(item)

    return results


class HybridSchemaLinker:
    """
    Hybrid retriever that combines BM25 and FAISS with RRF fusion.
    Serves two purposes:
      1. Schema linking: find relevant tables for a question
      2. Few-shot retrieval: find similar training examples
    """

    def __init__(self):
        self.bm25_index = None
        self.faiss_index = None

    def build(
        self,
        tables_path: str,
        train_jsonl_path: str,
        index_dir: str = "retrieval/indexes",
    ) -> None:
        """Build both BM25 and FAISS indexes."""
        os.makedirs(index_dir, exist_ok=True)

        # BM25 over schema
        console.print("\n[bold]Building BM25 index...[/bold]")
        from retrieval.bm25_index import BM25SchemaIndex
        with open(tables_path) as f:
            tables = json.load(f)
        self.bm25_index = BM25SchemaIndex()
        self.bm25_index.build(tables)
        self.bm25_index.save(os.path.join(index_dir, "bm25.pkl"))

        # FAISS over training examples
        console.print("\n[bold]Building FAISS index...[/bold]")
        try:
            from retrieval.schema_index import FAISSSchemaIndex
            with open(train_jsonl_path) as f:
                samples = [json.loads(l) for l in f if l.strip()]
            self.faiss_index = FAISSSchemaIndex()
            self.faiss_index.build(samples)
            self.faiss_index.save(os.path.join(index_dir, "faiss"))
        except ImportError:
            console.print("[yellow]FAISS not available — using BM25 only[/yellow]")

    @classmethod
    def load(cls, index_dir: str = "retrieval/indexes") -> "HybridSchemaLinker":
        """Load pre-built indexes."""
        instance = cls()

        bm25_path = os.path.join(index_dir, "bm25.pkl")
        if os.path.exists(bm25_path):
            from retrieval.bm25_index import BM25SchemaIndex
            instance.bm25_index = BM25SchemaIndex.load(bm25_path)

        faiss_dir = os.path.join(index_dir, "faiss")
        if os.path.exists(faiss_dir):
            try:
                from retrieval.schema_index import FAISSSchemaIndex
                instance.faiss_index = FAISSSchemaIndex.load(faiss_dir)
            except Exception as e:
                console.print(f"[yellow]FAISS load failed: {e}[/yellow]")

        return instance

    def get_relevant_tables(
        self,
        question: str,
        db_id: Optional[str] = None,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Retrieve top_k most relevant tables for a question.
        Uses RRF to fuse BM25 and FAISS rankings.
        """
        ranked_lists = []

        if self.bm25_index:
            bm25_results = self.bm25_index.search(question, db_id=db_id, top_k=top_k * 2)
            ranked_lists.append(bm25_results)

        if self.faiss_index:
            faiss_results = self.faiss_index.search(question, top_k=top_k * 2)
            ranked_lists.append(faiss_results)

        if not ranked_lists:
            return []

        if len(ranked_lists) == 1:
            return ranked_lists[0][:top_k]

        fused = reciprocal_rank_fusion(ranked_lists)
        return fused[:top_k]

    def get_few_shot_examples(
        self,
        question: str,
        top_k: int = 3,
    ) -> list[dict]:
        """
        Retrieve top_k most similar training examples for few-shot prompting.
        """
        if self.faiss_index:
            return self.faiss_index.search(question, top_k=top_k)
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--tables", default="dataset/spider/tables.json")
    parser.add_argument("--train", default="data/processed/train.jsonl")
    parser.add_argument("--index-dir", default="retrieval/indexes")
    parser.add_argument("--query", default=None)
    args = parser.parse_args()

    if args.build:
        linker = HybridSchemaLinker()
        linker.build(args.tables, args.train, args.index_dir)
        console.print("\n[bold green]✓ Hybrid index built successfully![/bold green]")

    if args.query:
        linker = HybridSchemaLinker.load(args.index_dir)
        tables = linker.get_relevant_tables(args.query, top_k=5)
        console.print(f"\n[bold]Top tables for:[/bold] {args.query}")
        for t in tables:
            console.print(f"  [{t.get('rrf_score', 0):.4f}] {t.get('db_id', '')} → {t.get('table_name', '')} | cols: {', '.join(t.get('columns', [])[:5])}")
