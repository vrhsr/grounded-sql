"""
data_pipeline.py — System 1: Data Pipeline
Transforms raw Spider dataset into Mistral-7B instruction format.

Run:
    python data_pipeline.py --config training/config.yaml

Output:
    data/processed/train.jsonl
    data/processed/validation.jsonl
    data/processed/test.jsonl
"""

import json
import os
import random
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Optional

import yaml
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

console = Console()

# ──────────────────────────────────────────────────────────────────
# Schema Builder — Converts tables.json → CREATE TABLE statements
# ──────────────────────────────────────────────────────────────────

TYPE_MAP = {
    "number": "REAL",
    "text": "TEXT",
    "time": "DATETIME",
    "boolean": "INTEGER",
    "others": "TEXT",
}


def build_create_statements(db_schema: dict) -> str:
    """
    Convert a Spider tables.json entry into CREATE TABLE SQL statements.
    Includes primary keys and foreign key references.
    """
    table_names = db_schema["table_names_original"]
    col_names = db_schema["column_names_original"]   # [(table_idx, col_name), ...]
    col_types = db_schema["column_types"]
    primary_keys = set(db_schema["primary_keys"])
    foreign_keys = {fk[0]: fk[1] for fk in db_schema["foreign_keys"]}

    # Group columns by table
    tables: dict[int, list] = defaultdict(list)
    for col_idx, (table_idx, col_name) in enumerate(col_names):
        if table_idx == -1:  # skip the wildcard (*) column
            continue
        tables[table_idx].append((col_idx, col_name, col_types[col_idx]))

    statements = []
    for table_idx, cols in tables.items():
        table_name = table_names[table_idx]
        lines = []
        for col_idx, col_name, col_type in cols:
            sql_type = TYPE_MAP.get(col_type, "TEXT")
            pk_marker = " PRIMARY KEY" if col_idx in primary_keys else ""
            lines.append(f"    {col_name} {sql_type}{pk_marker}")

        # Append FOREIGN KEY constraints
        for col_idx, col_name, _ in cols:
            if col_idx in foreign_keys:
                ref_col_idx = foreign_keys[col_idx]
                ref_table_idx, ref_col_name = col_names[ref_col_idx]
                ref_table_name = table_names[ref_table_idx]
                lines.append(
                    f"    FOREIGN KEY ({col_name}) REFERENCES {ref_table_name}({ref_col_name})"
                )

        stmt = f"CREATE TABLE {table_name} (\n" + ",\n".join(lines) + "\n);"
        statements.append(stmt)

    return "\n\n".join(statements)


# ──────────────────────────────────────────────────────────────────
# Mistral Instruction Template
# ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert SQL assistant. Given a database schema and a natural language question, "
    "generate the correct SQL query. Return only the SQL query, no explanations."
)


def format_instruction(schema_sql: str, question: str, gold_sql: str) -> dict:
    """
    Format a single (schema, question, SQL) triple into Mistral-7B-Instruct chat format.
    Returns a dict with 'prompt' (input) and 'completion' (expected output).
    """
    prompt = (
        f"[INST] {SYSTEM_PROMPT}\n\n"
        f"Database Schema:\n{schema_sql}\n\n"
        f"Question: {question} [/INST]"
    )
    return {
        "prompt": prompt,
        "completion": f" {gold_sql.strip()}",  # leading space is Mistral convention
        "text": f"{prompt} {gold_sql.strip()}</s>",  # full sequence for SFT
    }


# ──────────────────────────────────────────────────────────────────
# Core Pipeline
# ──────────────────────────────────────────────────────────────────

def load_schema_map(tables_path: str) -> dict[str, dict]:
    """Load tables.json → {db_id: schema_dict}"""
    with open(tables_path, "r") as f:
        tables = json.load(f)
    return {t["db_id"]: t for t in tables}


def process_split(
    data: list[dict],
    schema_map: dict[str, dict],
    max_seq_length: int = 512,
    tokenizer=None,
) -> tuple[list[dict], dict]:
    """
    Process a list of Spider samples into instruction format.
    Returns (processed_samples, stats).
    """
    processed = []
    stats = {
        "total": len(data),
        "processed": 0,
        "filtered_long": 0,
        "filtered_missing_schema": 0,
    }

    for item in tqdm(data, desc="Processing"):
        db_id = item["db_id"]
        question = item["question"]
        gold_sql = item["query"]

        if db_id not in schema_map:
            stats["filtered_missing_schema"] += 1
            continue

        schema_sql = build_create_statements(schema_map[db_id])
        sample = format_instruction(schema_sql, question, gold_sql)

        # Token-length filter (approximate without tokenizer)
        approx_tokens = len(sample["text"].split()) * 1.3
        if tokenizer:
            tokens = tokenizer(sample["text"], return_tensors="pt")
            n_tokens = tokens["input_ids"].shape[-1]
        else:
            n_tokens = int(approx_tokens)

        if n_tokens > max_seq_length:
            stats["filtered_long"] += 1
            continue

        sample["db_id"] = db_id
        sample["question"] = question
        sample["gold_sql"] = gold_sql
        processed.append(sample)
        stats["processed"] += 1

    return processed, stats


def save_jsonl(data: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    console.print(f"  [green]✓[/green] Saved {len(data):,} samples → {path}")


def print_stats_table(splits: dict[str, dict]) -> None:
    table = Table(title="Data Pipeline Statistics")
    table.add_column("Split", style="cyan")
    table.add_column("Total", justify="right")
    table.add_column("Processed", justify="right", style="green")
    table.add_column("Filtered (long)", justify="right", style="yellow")
    table.add_column("Missing schema", justify="right", style="red")
    table.add_column("Retention %", justify="right")

    for split_name, stats in splits.items():
        pct = 100 * stats["processed"] / max(stats["total"], 1)
        table.add_row(
            split_name,
            str(stats["total"]),
            str(stats["processed"]),
            str(stats["filtered_long"]),
            str(stats["filtered_missing_schema"]),
            f"{pct:.1f}%",
        )
    console.print(table)


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

def main(config_path: str = "training/config.yaml", dry_run: bool = False):
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]
    max_seq_len = cfg["model"]["max_seq_length"]

    console.rule("[bold blue]System 1 — Data Pipeline")

    # ── Load schema map ──────────────────────────────────────────
    console.print("\n[bold]Loading schema map from tables.json...[/bold]")
    schema_map = load_schema_map(data_cfg["tables"])
    console.print(f"  [green]✓[/green] Loaded {len(schema_map)} database schemas")

    # ── Load raw data ────────────────────────────────────────────
    console.print("\n[bold]Loading raw Spider data...[/bold]")
    with open(data_cfg["spider_train"]) as f:
        train_spider = json.load(f)
    with open(data_cfg["spider_others"]) as f:
        train_others = json.load(f)
    with open(data_cfg["spider_dev"]) as f:
        dev_data = json.load(f)

    # Combine train_spider + train_others
    all_train = train_spider + train_others
    random.seed(42)
    random.shuffle(all_train)

    # Hold out 500 from train as validation
    val_data = all_train[:500]
    train_data = all_train[500:]

    console.print(f"  Train: {len(train_data):,} | Val: {len(val_data):,} | Test: {len(dev_data):,}")

    # ── Process splits ────────────────────────────────────────────
    console.print("\n[bold]Processing train split...[/bold]")
    train_processed, train_stats = process_split(train_data, schema_map, max_seq_len)

    console.print("\n[bold]Processing validation split...[/bold]")
    val_processed, val_stats = process_split(val_data, schema_map, max_seq_len)

    console.print("\n[bold]Processing test split...[/bold]")
    test_processed, test_stats = process_split(dev_data, schema_map, max_seq_len)

    print_stats_table({
        "train": train_stats,
        "validation": val_stats,
        "test": test_stats,
    })

    # ── Mix in synthetic data if available ────────────────────────
    synthetic_path = data_cfg.get("synthetic_ecommerce")
    if synthetic_path and os.path.exists(synthetic_path):
        console.print(f"\n[bold]Loading synthetic e-commerce data...[/bold]")
        with open(synthetic_path) as f:
            synthetic = [json.loads(l) for l in f if l.strip()]
        train_processed.extend(synthetic)
        console.print(f"  [green]✓[/green] Added {len(synthetic)} synthetic samples")
        random.shuffle(train_processed)

    # ── Save ──────────────────────────────────────────────────────
    if not dry_run:
        console.print("\n[bold]Saving processed data...[/bold]")
        save_jsonl(train_processed, os.path.join(data_cfg["processed_dir"], "train.jsonl"))
        save_jsonl(val_processed, os.path.join(data_cfg["processed_dir"], "validation.jsonl"))
        save_jsonl(test_processed, os.path.join(data_cfg["processed_dir"], "test.jsonl"))

        # Save metadata for reproducibility
        metadata = {
            "train_count": len(train_processed),
            "val_count": len(val_processed),
            "test_count": len(test_processed),
            "max_seq_length": max_seq_len,
            "schema_dbs": len(schema_map),
            "seed": 42,
        }
        with open(os.path.join(data_cfg["processed_dir"], "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        console.print("\n[bold green]✓ Data pipeline complete![/bold green]")
    else:
        console.print("\n[yellow]Dry run — no files saved.[/yellow]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text-to-SQL Data Pipeline")
    parser.add_argument("--config", default="training/config.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Process but don't save")
    args = parser.parse_args()
    main(config_path=args.config, dry_run=args.dry_run)
