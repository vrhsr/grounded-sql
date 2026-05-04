"""
training/dataset.py — Hugging Face Dataset wrapper for fine-tuning
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class TextToSQLDataset(Dataset):
    """
    Dataset for Text-to-SQL instruction fine-tuning.
    Reads from pre-processed JSONL files produced by data_pipeline.py.
    """

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        prompt_key: str = "prompt",
        completion_key: str = "completion",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_key = prompt_key
        self.completion_key = completion_key

        self.samples = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        full_text = sample["text"]  # prompt + completion + </s>

        # Tokenize full sequence
        encoded = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)

        # Build labels: mask out the prompt tokens with -100
        # so loss is only computed on the completion (SQL) tokens
        prompt_encoded = self.tokenizer(
            sample[self.prompt_key],
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_len = prompt_encoded["input_ids"].shape[-1]

        labels = input_ids.clone()
        labels[:prompt_len] = -100  # ignore prompt in loss
        # Mask padding
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class DataCollatorForSQLSFT:
    """
    Custom data collator — pads sequences within a batch to the same length.
    More efficient than padding everything to max_length globally.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, padding_value: int = 0):
        self.tokenizer = tokenizer
        self.padding_value = padding_value

    def __call__(self, batch: list[dict]) -> dict:
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [b["input_ids"] for b in batch],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id or 0,
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [b["attention_mask"] for b in batch],
            batch_first=True,
            padding_value=0,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [b["labels"] for b in batch],
            batch_first=True,
            padding_value=-100,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
