import os

checks = [
    ("data/processed/train.jsonl", "Processed train data"),
    ("data/processed/validation.jsonl", "Processed validation data"),
    ("data/processed/test.jsonl", "Processed test data"),
    ("data/processed/metadata.json", "Metadata"),
    ("retrieval/indexes/bm25.pkl", "BM25 index"),
    ("retrieval/indexes/faiss/schema.faiss", "FAISS index"),
    ("checkpoints/rank_8/final_adapter", "Checkpoint rank-8"),
    ("checkpoints/rank_16/final_adapter", "Checkpoint rank-16"),
    ("checkpoints/rank_64/final_adapter", "Checkpoint rank-64"),
    ("evaluation/results/four_way_table.csv", "Four-way results"),
    ("evaluation/results/tanglish_results.csv", "Tanglish results"),
    ("data/raw/synthetic/ecommerce.jsonl", "Synthetic e-commerce data"),
    ("data/raw/synthetic/tanglish_test.jsonl", "Tanglish test data"),
]

print("=" * 60)
print("PROJECT STATUS CHECK")
print("=" * 60)
for path, label in checks:
    exists = os.path.exists(path)
    size = ""
    if exists:
        bytes_ = os.path.getsize(path)
        if bytes_ > 1024*1024:
            size = f" ({bytes_/1024/1024:.1f} MB)"
        else:
            size = f" ({bytes_/1024:.1f} KB)"
    status = "DONE" if exists else "TODO"
    print(f"  [{status}] {label:<35} {path}{size}")
