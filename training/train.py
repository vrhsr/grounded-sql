"""
training/train.py — System 2: QLoRA Fine-tuning Pipeline

Usage:
    # Standard rank-16 run
    python training/train.py --config training/config.yaml

    # Ablation runs
    python training/train.py --config training/config.yaml --override training/ablations/rank_8.yaml
    python training/train.py --config training/config.yaml --override training/ablations/rank_64.yaml
"""

import os
import sys
import argparse
from pathlib import Path

import yaml
import torch
import wandb
from rich.console import Console
from rich.panel import Panel

# ── Hugging Face + PEFT ────────────────────────────────────────
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# ── Local imports ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from training.dataset import TextToSQLDataset

console = Console()


# ──────────────────────────────────────────────────────────────────
# Config Loading
# ──────────────────────────────────────────────────────────────────

def load_config(base_path: str, override_path: Optional[str] = None) -> dict:
    """Load base config, then deep-merge override if provided."""
    with open(base_path) as f:
        cfg = yaml.safe_load(f)

    if override_path:
        with open(override_path) as f:
            override = yaml.safe_load(f)
        # Shallow merge at top level — override keys replace base keys
        for key, val in override.items():
            if isinstance(val, dict) and key in cfg:
                cfg[key].update(val)
            else:
                cfg[key] = val
        console.print(f"[yellow]Override applied: {override_path}[/yellow]")

    return cfg


# ──────────────────────────────────────────────────────────────────
# W&B GPU Memory Callback
# ──────────────────────────────────────────────────────────────────

class GPUMemoryCallback(TrainerCallback):
    """Logs GPU memory usage to W&B at each logging step."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if torch.cuda.is_available() and logs is not None:
            mem_allocated = torch.cuda.memory_allocated() / 1e9
            mem_reserved = torch.cuda.memory_reserved() / 1e9
            logs["system/gpu_memory_allocated_gb"] = mem_allocated
            logs["system/gpu_memory_reserved_gb"] = mem_reserved


# ──────────────────────────────────────────────────────────────────
# Model Setup
# ──────────────────────────────────────────────────────────────────

def build_model_and_tokenizer(cfg: dict):
    """Load base model with 4-bit quantization and attach LoRA adapters."""

    model_name = cfg["model"]["base_model"]
    q_cfg = cfg["quantization"]
    l_cfg = cfg["lora"]

    console.print(f"\n[bold]Loading base model:[/bold] {model_name}")

    # ── Tokenizer ────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"  # required for training

    # ── 4-bit Quantization ────────────────────────────────────────
    compute_dtype = (
        torch.bfloat16 if q_cfg["bnb_4bit_compute_dtype"] == "bfloat16" else torch.float16
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=q_cfg["load_in_4bit"],
        bnb_4bit_quant_type=q_cfg["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=q_cfg["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=compute_dtype,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=compute_dtype,
    )
    model = prepare_model_for_kbit_training(model)

    # ── LoRA Adapters ─────────────────────────────────────────────
    lora_config = LoraConfig(
        r=l_cfg["r"],
        lora_alpha=l_cfg["lora_alpha"],
        lora_dropout=l_cfg["lora_dropout"],
        bias=l_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
        target_modules=l_cfg["target_modules"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, tokenizer


# ──────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────

def train(cfg: dict) -> None:
    t_cfg = cfg["training"]
    d_cfg = cfg["data"]
    w_cfg = cfg["wandb"]

    # ── W&B Init ─────────────────────────────────────────────────
    run_name = Path(t_cfg["output_dir"]).name
    wandb.init(
        project=w_cfg["project"],
        entity=w_cfg.get("entity"),
        name=run_name,
        config={
            "lora_r": cfg["lora"]["r"],
            "lora_alpha": cfg["lora"]["lora_alpha"],
            "learning_rate": t_cfg["learning_rate"],
            "epochs": t_cfg["num_train_epochs"],
            "batch_size": t_cfg["per_device_train_batch_size"],
            "grad_accum": t_cfg["gradient_accumulation_steps"],
            "effective_batch": t_cfg["per_device_train_batch_size"] * t_cfg["gradient_accumulation_steps"],
        },
    )

    console.print(Panel(
        f"[bold]Run:[/bold] {run_name}\n"
        f"[bold]LoRA rank:[/bold] {cfg['lora']['r']}\n"
        f"[bold]LR:[/bold] {t_cfg['learning_rate']}\n"
        f"[bold]Epochs:[/bold] {t_cfg['num_train_epochs']}\n"
        f"[bold]Effective batch:[/bold] {t_cfg['per_device_train_batch_size'] * t_cfg['gradient_accumulation_steps']}",
        title="Training Configuration",
        border_style="blue",
    ))

    # ── Model + Tokenizer ─────────────────────────────────────────
    model, tokenizer = build_model_and_tokenizer(cfg)

    # ── Datasets ──────────────────────────────────────────────────
    max_len = cfg["model"]["max_seq_length"]
    train_dataset = TextToSQLDataset(
        os.path.join(d_cfg["processed_dir"], "train.jsonl"),
        tokenizer,
        max_length=max_len,
    )
    eval_dataset = TextToSQLDataset(
        os.path.join(d_cfg["processed_dir"], "validation.jsonl"),
        tokenizer,
        max_length=max_len,
    )
    console.print(f"\n[green]✓[/green] Train: {len(train_dataset):,} | Eval: {len(eval_dataset):,}")

    # ── Training Arguments ────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=t_cfg["output_dir"],
        num_train_epochs=t_cfg["num_train_epochs"],
        per_device_train_batch_size=t_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=t_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=t_cfg["gradient_accumulation_steps"],
        learning_rate=t_cfg["learning_rate"],
        lr_scheduler_type=t_cfg["lr_scheduler_type"],
        warmup_ratio=t_cfg["warmup_ratio"],
        max_grad_norm=t_cfg["max_grad_norm"],
        fp16=t_cfg["fp16"],
        bf16=t_cfg["bf16"],
        logging_steps=t_cfg["logging_steps"],
        evaluation_strategy="steps",
        eval_steps=t_cfg["eval_steps"],
        save_strategy="steps",
        save_steps=t_cfg["save_steps"],
        load_best_model_at_end=t_cfg["load_best_model_at_end"],
        metric_for_best_model=t_cfg["metric_for_best_model"],
        greater_is_better=False,
        report_to=t_cfg.get("report_to", "wandb"),
        dataloader_num_workers=t_cfg.get("dataloader_num_workers", 0),
        remove_unused_columns=t_cfg.get("remove_unused_columns", False),
        group_by_length=True,  # group similar-length sequences → faster training
        save_total_limit=3,
        prediction_loss_only=True,
    )

    # ── Trainer ──────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_seq_length=max_len,
        dataset_text_field="text",
        callbacks=[GPUMemoryCallback()],
    )

    # ── Train ─────────────────────────────────────────────────────
    console.print("\n[bold green]Starting training...[/bold green]")
    trainer.train()

    # ── Save final adapter ────────────────────────────────────────
    final_path = os.path.join(t_cfg["output_dir"], "final_adapter")
    trainer.model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    console.print(f"\n[bold green]✓ Training complete. Adapter saved to:[/bold green] {final_path}")

    wandb.finish()


# ──────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA Fine-tuning for Text-to-SQL")
    parser.add_argument("--config", default="training/config.yaml", help="Base config path")
    parser.add_argument("--override", default=None, help="Ablation override config path")
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    train(cfg)
