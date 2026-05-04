"""
retrieval/bm25_index.py — BM25 sparse index over column names

BM25 excels at exact keyword matching — critical for schema linking
because column names are literal tokens (e.g., "customer_id", "created_at").

Usage:
    python retrieval/bm25_index.py --build --tables dataset/spider/tables.json
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path

from rich.console import Console

console = Console()

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


class BM25SchemaIndex:
    """
    Sparse BM25 index over column and table names.
    Tokenizes column names into subwords (snake_case split) for better matching.
    """

    def __init__(self):
        if not BM25_AVAILABLE:
            raise ImportError("Install rank_bm25: pip install rank-bm25")
        self.bm25 = None
        self.corpus_metadata = []  # list of {db_id, table_name, columns, full_text}

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """
        Tokenize schema text. Splits snake_case and camelCase.
        'customer_id' → ['customer', 'id']
        'orderValue' → ['order', 'value']
        """
        import re
        # Split snake_case
        text = text.replace("_", " ")
        # Split camelCase
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        return text.lower().split()

    def build(self, tables: list[dict]) -> None:
        """Build BM25 index over all schemas in tables.json."""
        console.print(f"\n[bold]Building BM25 index over {len(tables)} databases...[/bold]")

        corpus = []
        for db in tables:
            db_id = db["db_id"]
            table_names = db["table_names_original"]
            col_names = db["column_names_original"]

            # Group columns by table
            for t_idx, t_name in enumerate(table_names):
                cols = [c for (ti, c) in col_names if ti == t_idx]
                full_text = f"{db_id} {t_name} " + " ".join(cols)
                tokens = self.tokenize(full_text)
                corpus.append(tokens)
                self.corpus_metadata.append({
                    "db_id": db_id,
                    "table_name": t_name,
                    "columns": cols,
                    "full_text": full_text,
                })

        self.bm25 = BM25Okapi(corpus)
        console.print(f"[green]✓[/green] BM25 index built: {len(corpus)} table entries")

    def search(self, query: str, db_id: str = None, top_k: int = 5) -> list[dict]:
        """
        Search for relevant tables/columns for a query.
        Optionally filter to a specific database.
        """
        if self.bm25 is None:
            return []

        tokens = self.tokenize(query)
        scores = self.bm25.get_scores(tokens)

        # Get top_k indices
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in ranked:
            meta = self.corpus_metadata[idx]
            if db_id and meta["db_id"] != db_id:
                continue
            if score > 0:
                r = dict(meta)
                r["score"] = float(score)
                results.append(r)
            if len(results) >= top_k:
                break

        return results

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "metadata": self.corpus_metadata}, f)
        console.print(f"[green]✓[/green] BM25 index saved to {path}")

    @classmethod
    def load(cls, path: str) -> "BM25SchemaIndex":
        instance = cls.__new__(cls)
        with open(path, "rb") as f:
            data = pickle.load(f)
        instance.bm25 = data["bm25"]
        instance.corpus_metadata = data["metadata"]
        console.print(f"[green]✓[/green] BM25 index loaded: {len(instance.corpus_metadata)} entries")
        return instance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--tables", default="dataset/spider/tables.json")
    parser.add_argument("--index-path", default="retrieval/indexes/bm25.pkl")
    parser.add_argument("--query", default=None)
    parser.add_argument("--db-id", default=None)
    args = parser.parse_args()

    if args.build:
        with open(args.tables) as f:
            tables = json.load(f)
        idx = BM25SchemaIndex()
        idx.build(tables)
        idx.save(args.index_path)

    if args.query:
        idx = BM25SchemaIndex.load(args.index_path)
        results = idx.search(args.query, db_id=args.db_id, top_k=5)
        for r in results:
            console.print(f"\nScore: {r['score']:.3f} | DB: {r['db_id']} | Table: {r['table_name']}")
            console.print(f"  Columns: {', '.join(r['columns'][:8])}")
