"""serving/model_loader.py — Load and manage the INT8-quantized inference model"""

import os
import sys
import asyncio
from pathlib import Path
from typing import AsyncGenerator, Optional

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from threading import Thread


SYSTEM_PROMPT = (
    "You are an expert SQL assistant. Given a database schema and a natural language question, "
    "generate the correct SQL query. Return only the SQL query, no explanations."
)


class ModelLoader:
    """
    Manages loading and inference of the fine-tuned Mistral-7B model.
    Supports INT8 quantization for production serving.
    """

    def __init__(
        self,
        adapter_path: str,
        base_model: str = "mistralai/Mistral-7B-Instruct-v0.2",
        use_int8: bool = True,
    ):
        self.adapter_path = adapter_path
        self.base_model = base_model
        self.use_int8 = use_int8
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
     
    def load(self) -> None:
        """Load model with INT8 quantization for production serving."""
        print(f"Loading model from {self.adapter_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.adapter_path if os.path.exists(self.adapter_path) else self.base_model,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.use_int8 and torch.cuda.is_available():
            # INT8 quantization — 2.3x throughput, <1% accuracy drop
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            model_path = self.adapter_path if os.path.exists(self.adapter_path) else self.base_model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            # CPU fallback — slower but works without GPU
            model_path = self.adapter_path if os.path.exists(self.adapter_path) else self.base_model
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="cpu",
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )

        self.model.eval()
        self.is_loaded = True
        print("✓ Model loaded")

    def _build_prompt(self, question: str, schema: str) -> str:
        return (
            f"[INST] {SYSTEM_PROMPT}\n\n"
            f"Database Schema:\n{schema}\n\n"
            f"Question: {question} [/INST]"
        )

    @torch.no_grad()
    def generate(
        self,
        question: str,
        schema: str,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
    ) -> str:
        """Synchronous SQL generation for use with run_in_executor."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
    
        prompt = self._build_prompt(question, schema)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    async def generate_stream(
        self,
        question: str,
        schema: str,
        max_new_tokens: int = 256,
    ) -> AsyncGenerator[str, None]:
        """Async streaming generation using TextIteratorStreamer."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        prompt = self._build_prompt(question, schema)
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.model.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "streamer": streamer,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        # Run generation in a thread to avoid blocking the event loop
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        for token in streamer:
            yield token
            await asyncio.sleep(0)  # yield control back to event loop

        thread.join()
