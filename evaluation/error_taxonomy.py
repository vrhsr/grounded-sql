"""
evaluation/error_taxonomy.py — Categorize and analyze failure modes.

Reads raw results CSVs from four_way_compare.py and produces
the error breakdown table used in the README and interviews.

Usage:
    python evaluation/error_taxonomy.py --results evaluation/results/c:_fine-tuned_results.csv
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path
from collections import defaultdict

import pandas as pd
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))

console = Console()

# ──────────────────────────────────────────────────────────────────
# Error Taxonomy Categories
# ──────────────────────────────────────────────────────────────────

TAXONOMY = {
    "hallucinated_column": "SQL references a column that doesn't exist in schema",
    "wrong_join":          "Incorrect JOIN type or missing/wrong JOIN condition",
    "wrong_aggregation":   "Wrong aggregate function or missing GROUP BY",
    "wrong_where":         "Incorrect WHERE clause (operator, value, or column)",
    "syntax_error":        "SQL failed to parse/execute (syntax error)",
    "correct_sql_wrong_data": "SQL is semantically valid but returns wrong rows",
}


def detect_hallucinated_column(pred_sql: str, schema_cols: set) -> bool:
    """Heuristic: check if pred_sql references any column not in the schema."""
    # Extract identifiers that look like column names
    tokens = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', pred_sql)
    # SQL keywords to ignore
    sql_keywords = {
        "SELECT", "FROM", "WHERE", "JOIN", "ON", "AS", "AND", "OR", "NOT",
        "IN", "LIKE", "BETWEEN", "IS", "NULL", "GROUP", "BY", "HAVING",
        "ORDER", "LIMIT", "DISTINCT", "COUNT", "SUM", "AVG", "MAX", "MIN",
        "INNER", "LEFT", "RIGHT", "OUTER", "CROSS", "UNION", "INTERSECT",
        "EXCEPT", "CASE", "WHEN", "THEN", "ELSE", "END", "WITH", "RECURSIVE",
        "INSERT", "UPDATE", "DELETE", "CREATE", "TABLE", "INDEX", "VIEW",
    }
    tokens = {t.upper() for t in tokens if t.upper() not in sql_keywords}
    # If any token isn't in known columns (case-insensitive), flag it
    schema_upper = {c.upper() for c in schema_cols}
    hallucinated = tokens - schema_upper
    # Filter out numbers and common words
    hallucinated = {t for t in hallucinated if not t.isdigit() and len(t) > 2}
    return len(hallucinated) > 0


def categorize_error(row: pd.Series, schema_cols: set = None) -> str:
    """
    Categorize the error type for a single wrong prediction.
    Uses heuristics on the SQL structure.
    """
    pred_sql = str(row.get("pred_sql", "")).upper()
    error_type = str(row.get("error_type", ""))

    if error_type == "execution_error":
        return "syntax_error"

    if schema_cols and detect_hallucinated_column(str(row.get("pred_sql", "")), schema_cols):
        return "hallucinated_column"

    # JOIN-related patterns
    gold_sql = str(row.get("gold_sql", "")).upper()
    pred_joins = pred_sql.count(" JOIN ")
    gold_joins = gold_sql.count(" JOIN ")
    if abs(pred_joins - gold_joins) > 0:
        return "wrong_join"
    if "LEFT JOIN" in gold_sql and "LEFT JOIN" not in pred_sql:
        return "wrong_join"

    # Aggregation patterns
    agg_fns = ["COUNT(", "SUM(", "AVG(", "MAX(", "MIN("]
    pred_has_agg = any(fn in pred_sql for fn in agg_fns)
    gold_has_agg = any(fn in gold_sql for fn in agg_fns)
    if pred_has_agg != gold_has_agg:
        return "wrong_aggregation"
    if "GROUP BY" in gold_sql and "GROUP BY" not in pred_sql:
        return "wrong_aggregation"

    # WHERE clause patterns
    if "WHERE" in gold_sql and "WHERE" not in pred_sql:
        return "wrong_where"

    return "correct_sql_wrong_data"


def analyze_errors(df: pd.DataFrame, schema_cols: set = None) -> pd.DataFrame:
    """Add error category column to wrong predictions dataframe."""
    wrong = df[df["correct"] == False].copy()
    if len(wrong) == 0:
        console.print("[green]No errors found! Perfect model.[/green]")
        return wrong

    wrong["error_category"] = wrong.apply(
        lambda row: categorize_error(row, schema_cols), axis=1
    )
    return wrong


def print_taxonomy_table(wrong_df: pd.DataFrame, system_name: str = "System") -> None:
    if len(wrong_df) == 0:
        return

    counts = wrong_df["error_category"].value_counts()
    total_errors = len(wrong_df)

    table = Table(title=f"Error Taxonomy — {system_name}", show_lines=True)
    table.add_column("Error Category", style="cyan")
    table.add_column("Description", style="dim", width=40)
    table.add_column("Count", justify="right")
    table.add_column("% of Errors", justify="right", style="yellow")

    for cat, desc in TAXONOMY.items():
        count = counts.get(cat, 0)
        pct = 100 * count / total_errors if total_errors > 0 else 0
        table.add_row(cat, desc, str(count), f"{pct:.1f}%")

    console.print(table)
    console.print(f"\nTotal errors: {total_errors} / {len(wrong_df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to system results CSV")
    parser.add_argument("--system-name", default="Fine-tuned Model")
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    wrong_df = analyze_errors(df)
    print_taxonomy_table(wrong_df, system_name=args.system_name)

    # Save breakdown
    out_path = args.results.replace(".csv", "_error_breakdown.csv")
    if len(wrong_df) > 0:
        wrong_df[["question", "db_id", "pred_sql", "gold_sql", "error_category"]].to_csv(out_path, index=False)
        console.print(f"[green]✓[/green] Error breakdown saved to {out_path}")
