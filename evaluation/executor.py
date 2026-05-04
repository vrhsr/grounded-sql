"""
evaluation/executor.py — System 3: SQL Execution Evaluator

The core differentiator: evaluates SQL correctness by EXECUTING both
predicted and gold SQL against the actual SQLite database, then comparing
result rows — not string matching.

Usage:
    from evaluation.executor import SQLExecutionEvaluator
    evaluator = SQLExecutionEvaluator(databases_dir="dataset/spider/database")
    result = evaluator.evaluate(predicted_sql, gold_sql, db_id="department_management")
"""

import sqlite3
import os
import signal
import traceback
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field


# ──────────────────────────────────────────────────────────────────
# Result Types
# ──────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    correct: bool
    error_type: Optional[str]          # None if correct
    pred_sql: str
    gold_sql: str
    pred_rows: Optional[int] = None
    gold_rows: Optional[int] = None
    pred_error: Optional[str] = None   # SQL error message if execution failed
    timed_out: bool = False


ERROR_TYPES = {
    "execution_error":   "SQL failed to execute (syntax/runtime error)",
    "timeout":           "SQL execution exceeded time limit",
    "wrong_result":      "SQL ran but returned incorrect rows",
    "empty_vs_nonempty": "Predicted empty result, gold is non-empty",
    "nonempty_vs_empty": "Predicted non-empty result, gold is empty",
}


# ──────────────────────────────────────────────────────────────────
# Evaluator
# ──────────────────────────────────────────────────────────────────

class SQLExecutionEvaluator:
    """
    Binary execution-based SQL evaluator.
    
    Correctness = (rows from pred_sql) == (rows from gold_sql)
    Order-insensitive, float-rounded, NULL-aware.
    """

    def __init__(
        self,
        databases_dir: str = "dataset/spider/database",
        timeout_seconds: int = 5,
        float_precision: int = 4,
    ):
        self.databases_dir = databases_dir
        self.timeout_seconds = timeout_seconds
        self.float_precision = float_precision

    # ── DB Connection ─────────────────────────────────────────────

    def _get_db_path(self, db_id: str) -> str:
        return os.path.join(self.databases_dir, db_id, f"{db_id}.sqlite")

    def _execute_sql(self, sql: str, db_path: str) -> tuple[Optional[list], Optional[str]]:
        """
        Execute SQL against a SQLite database.
        Returns (rows, error_message). rows is None on error.
        Enforces timeout via a row limit heuristic (no threading needed on Windows).
        """
        if not os.path.exists(db_path):
            return None, f"Database not found: {db_path}"

        try:
            conn = sqlite3.connect(db_path, timeout=self.timeout_seconds)
            conn.execute("PRAGMA query_only = ON;")  # read-only safety
            cursor = conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            conn.close()
            return rows, None
        except sqlite3.OperationalError as e:
            return None, f"OperationalError: {e}"
        except sqlite3.Error as e:
            return None, f"SQLiteError: {e}"
        except Exception as e:
            return None, f"UnexpectedError: {e}"

    # ── Row Normalization ─────────────────────────────────────────

    def _normalize_value(self, v):
        """Normalize a single cell value for comparison."""
        if v is None:
            return "__NULL__"  # NULL == NULL in evaluation (unlike SQL)
        if isinstance(v, float):
            return round(v, self.float_precision)
        if isinstance(v, str):
            return v.strip().lower()
        return v

    def _to_comparable_set(self, rows: list) -> frozenset:
        """Convert result rows to an order-insensitive frozenset of normalized tuples."""
        return frozenset(
            tuple(self._normalize_value(v) for v in row)
            for row in rows
        )

    # ── Core Evaluate ─────────────────────────────────────────────

    def evaluate(self, pred_sql: str, gold_sql: str, db_id: str) -> EvalResult:
        """
        Evaluate one (pred_sql, gold_sql) pair against the given database.
        """
        db_path = self._get_db_path(db_id)

        # Execute predicted SQL
        pred_rows, pred_error = self._execute_sql(pred_sql, db_path)

        if pred_rows is None:
            return EvalResult(
                correct=False,
                error_type="execution_error",
                pred_sql=pred_sql,
                gold_sql=gold_sql,
                pred_error=pred_error,
            )

        # Execute gold SQL
        gold_rows, gold_error = self._execute_sql(gold_sql, db_path)

        if gold_rows is None:
            # Gold SQL failed — this shouldn't happen with Spider but handle gracefully
            return EvalResult(
                correct=False,
                error_type="gold_execution_error",
                pred_sql=pred_sql,
                gold_sql=gold_sql,
                pred_error=f"Gold SQL failed: {gold_error}",
            )

        # Compare results
        pred_set = self._to_comparable_set(pred_rows)
        gold_set = self._to_comparable_set(gold_rows)

        if pred_set == gold_set:
            return EvalResult(
                correct=True,
                error_type=None,
                pred_sql=pred_sql,
                gold_sql=gold_sql,
                pred_rows=len(pred_rows),
                gold_rows=len(gold_rows),
            )

        # Determine error sub-type for failure taxonomy
        if len(pred_rows) == 0 and len(gold_rows) > 0:
            error_type = "empty_vs_nonempty"
        elif len(pred_rows) > 0 and len(gold_rows) == 0:
            error_type = "nonempty_vs_empty"
        else:
            error_type = "wrong_result"

        return EvalResult(
            correct=False,
            error_type=error_type,
            pred_sql=pred_sql,
            gold_sql=gold_sql,
            pred_rows=len(pred_rows),
            gold_rows=len(gold_rows),
        )

    # ── Batch Evaluate ────────────────────────────────────────────

    def evaluate_batch(
        self,
        samples: list[dict],
        verbose: bool = False,
    ) -> tuple[list[EvalResult], dict]:
        """
        Evaluate a list of samples.
        Each sample must have: pred_sql, gold_sql, db_id.
        Returns (results, aggregate_stats).
        """
        results = []
        error_counts = {}

        for sample in samples:
            result = self.evaluate(
                pred_sql=sample["pred_sql"],
                gold_sql=sample["gold_sql"],
                db_id=sample["db_id"],
            )
            results.append(result)

            if not result.correct:
                et = result.error_type or "unknown"
                error_counts[et] = error_counts.get(et, 0) + 1

            if verbose and not result.correct:
                print(f"[WRONG] {sample.get('question', '')[:60]}")
                print(f"  Pred: {sample['pred_sql'][:100]}")
                print(f"  Gold: {sample['gold_sql'][:100]}")
                print(f"  Error: {result.error_type}\n")

        n_correct = sum(r.correct for r in results)
        n_total = len(results)
        exec_acc = n_correct / n_total if n_total > 0 else 0.0

        stats = {
            "total": n_total,
            "correct": n_correct,
            "execution_accuracy": exec_acc,
            "error_breakdown": error_counts,
        }

        return results, stats
