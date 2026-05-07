"""
evaluation/four_way_compare.py — System 3: Four-Way Evaluation

Evaluates four systems on the same 1034 test queries:
  A: Base Mistral-7B-Instruct (no fine-tuning, no RAG)
  B: RAG only (few-shot with retrieved examples, no fine-tuning)
  C: Fine-tuned only (QLoRA, no RAG)
  D: Fine-tuned + RAG (QLoRA + schema retrieval)

Usage:
    python evaluation/four_way_compare.py \
        --config training/config.yaml \
        --finetuned-adapter checkpoints/rank_16/final_adapter \
        --systems A B C D
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.executor import SQLExecutionEvaluator

console = Console()


# ──────────────────────────────────────────────────────────────────
# SQL Generator — wraps model inference
# ──────────────────────────────────────────────────────────────────

class SQLGenerator:
    """Unified SQL generator supporting base and fine-tuned models."""

    SYSTEM_PROMPT = (
        "You are an expert SQL assistant. Given a database schema and a natural language question, "
        "generate the correct SQL query. Return only the SQL query, no explanations."
    )

    def __init__(self, model_path: str, use_4bit: bool = True):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel
        import os

        console.print(f"\n[bold]Loading model:[/bold] {model_path}")

        # Detect whether this is a PEFT adapter directory or a full model
        is_peft = os.path.exists(os.path.join(model_path, "adapter_config.json"))

        if is_peft:
            # Load the base model name from the adapter config
            import json
            with open(os.path.join(model_path, "adapter_config.json")) as f:
                adapter_cfg = json.load(f)
            base_model_name = adapter_cfg["base_model_name_or_path"]
            console.print(f"[dim]PEFT adapter detected — loading base: {base_model_name}[/dim]")
        else:
            base_model_name = model_path

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
            )

        # Apply LoRA adapter weights on top of the base model
        if is_peft:
            self.model = PeftModel.from_pretrained(self.model, model_path)
            self.model = self.model.merge_and_unload()  # merge for faster inference
            console.print("[green]✓ LoRA adapter merged[/green]")

        self.model.eval()

    def build_prompt(self, schema_sql: str, question: str, few_shot_examples: list[dict] = None) -> str:
        """Build the Mistral-7B-Instruct formatted prompt."""
        parts = [f"[INST] {self.SYSTEM_PROMPT}\n"]

        # Inject few-shot examples if provided (RAG)
        if few_shot_examples:
            parts.append("Here are some relevant examples:\n")
            for ex in few_shot_examples[:3]:  # max 3 examples
                parts.append(f"Schema: {ex['schema'][:300]}...")
                parts.append(f"Question: {ex['question']}")
                parts.append(f"SQL: {ex['sql']}\n")

        parts.append(f"Database Schema:\n{schema_sql}\n")
        parts.append(f"Question: {question} [/INST]")
        return "\n".join(parts)

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 256,
        temperature: float = 0.1,
    ) -> tuple[list[str], float]:
        """Generate SQL in batches. Returns (list of sqls, total_latency)."""
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(self.model.device)

        t0 = time.perf_counter()
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        latency = time.perf_counter() - t0

        generated_sqls = []
        for i, output in enumerate(outputs):
            gen_tokens = output[inputs["input_ids"].shape[1]:]
            sql = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
            generated_sqls.append(sql)
            
        return generated_sqls, latency

    @torch.no_grad()
    def generate(
        self,
        schema_sql: str,
        question: str,
        few_shot_examples: list[dict] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
    ) -> tuple[str, float]:
        """Generate SQL. Returns (sql, latency_seconds)."""
        prompt = self.build_prompt(schema_sql, question, few_shot_examples)
        sqls, lat = self.generate_batch([prompt], max_new_tokens, temperature)
        return sqls[0], lat


# ──────────────────────────────────────────────────────────────────
# RAG Retriever (stub — full impl in retrieval/hybrid_linker.py)
# ──────────────────────────────────────────────────────────────────

class RAGRetriever:
    """
    Retrieves semantically similar few-shot examples using the pre-built
    FAISS index (retrieval/indexes/faiss/) via HybridSchemaLinker.
    """

    def __init__(self, index_dir: str = "retrieval/indexes", train_data_path: str = "data/processed/train.jsonl"):
        self.linker = None
        try:
            from retrieval.hybrid_linker import HybridSchemaLinker
            self.linker = HybridSchemaLinker.load(index_dir)
            console.print(f"[green]✓ RAG: Loaded FAISS index from {index_dir}[/green]")
        except Exception as e:
            console.print(f"[yellow]Warning: FAISS index load failed ({e}). RAG will be disabled.[/yellow]")

    def retrieve(self, question: str, schema: str, top_k: int = 3) -> list[dict]:
        """Retrieve top_k semantically similar examples from the FAISS index."""
        if self.linker is None:
            return []
        try:
            results = self.linker.get_few_shot_examples(question, top_k=top_k)
            # Map FAISS metadata fields → prompt builder format
            return [
                {
                    "schema": r.get("schema_snippet", ""),
                    "question": r.get("question", ""),
                    "sql": r.get("gold_sql", ""),
                }
                for r in results
                if r.get("gold_sql")
            ]
        except Exception as e:
            console.print(f"[yellow]Retrieval failed: {e}[/yellow]")
            return []


# ──────────────────────────────────────────────────────────────────
# Four-Way Evaluation
# ──────────────────────────────────────────────────────────────────

def load_test_data(test_path: str, schema_map: dict) -> list[dict]:
    """Load test JSONL and attach schema strings."""
    samples = []
    with open(test_path) as f:
        for line in f:
            if line.strip():
                s = json.loads(line)
                samples.append(s)
    return samples


def run_system_evaluation(
    system_name: str,
    generator: SQLGenerator,
    test_samples: list[dict],
    evaluator: SQLExecutionEvaluator,
    rag_retriever: Optional[RAGRetriever] = None,
    use_rag: bool = False,
) -> dict:
    """Run one system on all test samples and return aggregate results."""

    results = []
    latencies = []
    errors = {"execution_error": 0, "wrong_result": 0, "timeout": 0,
              "empty_vs_nonempty": 0, "nonempty_vs_empty": 0}

    console.print(f"\n[bold blue]━━ System {system_name} ━━[/bold blue]")

    batch_size = 12
    
    for i in tqdm(range(0, len(test_samples), batch_size), desc=f"System {system_name}"):
        batch_samples = test_samples[i:i+batch_size]
        prompts = []
        
        for sample in batch_samples:
            schema_sql = sample.get("prompt", "").split("Database Schema:")[-1].split("Question:")[0].strip()
            question = sample.get("question", "")
            few_shots = rag_retriever.retrieve(question, schema_sql) if use_rag and rag_retriever else None
            prompts.append(generator.build_prompt(schema_sql, question, few_shots))
            
        pred_sqls, latency = generator.generate_batch(prompts)
        
        # Distribute latency equally for stats
        per_query_lat = latency / len(batch_samples)
        
        for sample, pred_sql in zip(batch_samples, pred_sqls):
            question = sample.get("question", "")
            gold_sql = sample.get("gold_sql", "")
            db_id = sample.get("db_id", "")
            
            latencies.append(per_query_lat)
            eval_result = evaluator.evaluate(pred_sql, gold_sql, db_id)
            results.append({
                "question": question,
                "db_id": db_id,
                "pred_sql": pred_sql,
                "gold_sql": gold_sql,
                "correct": eval_result.correct,
                "error_type": eval_result.error_type,
                "latency_s": per_query_lat,
            })

            if eval_result.error_type:
                errors[eval_result.error_type] = errors.get(eval_result.error_type, 0) + 1

    n_correct = sum(r["correct"] for r in results)
    n_total = len(results)
    exec_acc = n_correct / n_total

    # Latency percentiles
    sorted_lat = sorted(latencies)
    p50 = sorted_lat[int(0.50 * len(sorted_lat))]
    p95 = sorted_lat[int(0.95 * len(sorted_lat))]

    return {
        "system": system_name,
        "exec_accuracy": exec_acc,
        "n_correct": n_correct,
        "n_total": n_total,
        "p50_latency_s": p50,
        "p95_latency_s": p95,
        "error_breakdown": errors,
        "raw_results": results,
    }


def print_comparison_table(system_results: list[dict]) -> None:
    table = Table(title="Four-Way Comparison — Execution Accuracy", show_lines=True)
    table.add_column("System", style="cyan", width=20)
    table.add_column("Exec Accuracy", justify="right", style="bold green")
    table.add_column("Correct / Total", justify="right")
    table.add_column("p50 Latency", justify="right")
    table.add_column("p95 Latency", justify="right")

    for r in system_results:
        table.add_row(
            r["system"],
            f"{r['exec_accuracy']:.1%}",
            f"{r['n_correct']}/{r['n_total']}",
            f"{r['p50_latency_s']:.2f}s",
            f"{r['p95_latency_s']:.2f}s",
        )
    console.print(table)


def save_results(system_results: list[dict], output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # Summary CSV
    summary = [{
        "system": r["system"],
        "exec_accuracy": r["exec_accuracy"],
        "n_correct": r["n_correct"],
        "n_total": r["n_total"],
        "p50_latency_s": r["p50_latency_s"],
        "p95_latency_s": r["p95_latency_s"],
    } for r in system_results]
    pd.DataFrame(summary).to_csv(os.path.join(output_dir, "four_way_table.csv"), index=False)

    # Per-system detailed results
    for r in system_results:
        system_name = r["system"].replace(" ", "_").lower()
        pd.DataFrame(r["raw_results"]).to_csv(
            os.path.join(output_dir, f"{system_name}_results.csv"), index=False
        )

    console.print(f"\n[green]✓[/green] Results saved to {output_dir}/")


# ──────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Four-Way System Comparison")
    parser.add_argument("--config", default="training/config.yaml")
    parser.add_argument("--finetuned-adapter", default=None, help="Path to fine-tuned LoRA adapter")
    parser.add_argument("--systems", nargs="+", default=["A"], choices=["A", "B", "C", "D"])
    parser.add_argument("--max-samples", type=int, default=None, help="Limit test samples (for debugging)")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    d_cfg = cfg["data"]
    BASE_MODEL = cfg["model"]["base_model"]

    # Load test data
    test_path = os.path.join(d_cfg["processed_dir"], "test.jsonl")
    evaluator = SQLExecutionEvaluator(databases_dir=d_cfg["databases_dir"])

    test_samples = load_test_data(test_path, {})
    if args.max_samples:
        test_samples = test_samples[:args.max_samples]
    console.print(f"Evaluating on {len(test_samples)} test samples")

    system_results = []

    if "A" in args.systems or "B" in args.systems:
        base_gen = SQLGenerator(BASE_MODEL, use_4bit=True)
        rag = RAGRetriever(index_dir="retrieval/indexes")

    if "A" in args.systems:
        result_a = run_system_evaluation("A: Base Model", base_gen, test_samples, evaluator, use_rag=False)
        system_results.append(result_a)
        save_results(system_results, "evaluation/results")
        console.print("[green]✓ System A saved[/green]")

    if "B" in args.systems:
        result_b = run_system_evaluation("B: RAG Only", base_gen, test_samples, evaluator, rag_retriever=rag, use_rag=True)
        system_results.append(result_b)
        save_results(system_results, "evaluation/results")
        console.print("[green]✓ System B saved[/green]")

    if "C" in args.systems or "D" in args.systems:
        if not args.finetuned_adapter:
            console.print("[red]Error: --finetuned-adapter required for systems C and D[/red]")
            sys.exit(1)
        # Free VRAM before loading the fine-tuned model
        if "A" in args.systems or "B" in args.systems:
            console.print("\n[yellow]Freeing base model from VRAM...[/yellow]")
            del base_gen
            if "B" in args.systems:
                del rag
            torch.cuda.empty_cache()
            import gc; gc.collect()
            console.print("[green]✓ VRAM freed[/green]")
        ft_gen = SQLGenerator(args.finetuned_adapter, use_4bit=True)

    if "C" in args.systems:
        result_c = run_system_evaluation("C: Fine-tuned", ft_gen, test_samples, evaluator, use_rag=False)
        system_results.append(result_c)
        save_results(system_results, "evaluation/results")
        console.print("[green]✓ System C saved[/green]")

    if "D" in args.systems:
        rag_ft = RAGRetriever(index_dir="retrieval/indexes")
        result_d = run_system_evaluation("D: Fine-tuned + RAG", ft_gen, test_samples, evaluator, rag_retriever=rag_ft, use_rag=True)
        system_results.append(result_d)
    print_comparison_table(system_results)
    save_results(system_results, "evaluation/results")
