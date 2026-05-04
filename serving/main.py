"""
serving/main.py — System 4: FastAPI Serving Layer

Exposes your fine-tuned model as a REST API with:
  - Redis caching (cache_key = hash(question + schema))
  - Hybrid schema linking
  - Streaming support
  - Latency tracking

Start server:
    uvicorn serving.main:app --host 0.0.0.0 --port 8000 --reload

Test:
    curl -X POST http://localhost:8000/generate-sql \
      -H "Content-Type: application/json" \
      -d '{"question": "How many customers signed up last month?", "schema": "CREATE TABLE customers (customer_id INT, created_at DATETIME);"}'
"""

import os
import sys
import time
import hashlib
import json
import asyncio
from pathlib import Path
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).parent.parent))
from serving.schemas import SQLRequest, SQLResponse, HealthResponse
from serving.model_loader import ModelLoader
from serving.cache import RedisCache

# ──────────────────────────────────────────────────────────────────
# App State — loaded at startup
# ──────────────────────────────────────────────────────────────────

class AppState:
    model_loader: Optional[ModelLoader] = None
    cache: Optional[RedisCache] = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and connect cache at startup."""
    print("Starting Text-to-SQL API...")

    # Load model
    adapter_path = os.getenv("MODEL_ADAPTER_PATH", "checkpoints/rank_16/final_adapter")
    base_model = os.getenv("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
    state.model_loader = ModelLoader(adapter_path=adapter_path, base_model=base_model)
    state.model_loader.load()

    # Connect Redis cache
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    state.cache = RedisCache(url=redis_url)
    await state.cache.connect()

    print("✓ API ready")
    yield

    # Cleanup
    if state.cache:
        await state.cache.disconnect()


# ──────────────────────────────────────────────────────────────────
# FastAPI App
# ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Text-to-SQL API",
    description="Fine-tuned Mistral-7B for natural language to SQL conversion",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────
# Cache Key
# ──────────────────────────────────────────────────────────────────

def make_cache_key(question: str, schema: str) -> str:
    """Deterministic cache key from question + schema."""
    content = f"{question.strip().lower()}::{schema.strip()}"
    return hashlib.sha256(content.encode()).hexdigest()


# ──────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    gpu_mem = None
    if torch.cuda.is_available():
        gpu_mem = f"{torch.cuda.memory_allocated() / 1e9:.2f}GB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"

    return HealthResponse(
        status="healthy",
        model_loaded=state.model_loader is not None and state.model_loader.is_loaded,
        gpu_memory=gpu_mem,
    )


@app.post("/generate-sql", response_model=SQLResponse)
async def generate_sql(request: SQLRequest):
    """
    Generate SQL from natural language question + schema.
    
    Flow:
      1. Cache check (Redis)
      2. Schema linking (hybrid BM25+FAISS)
      3. SQL generation (INT8 quantized model)
      4. Cache write
      5. Return SQL + metadata
    """
    if state.model_loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t_start = time.perf_counter()

    # ── Step 1: Cache check ──────────────────────────────────────
    cache_key = make_cache_key(request.question, request.schema)
    if state.cache:
        cached = await state.cache.get(cache_key)
        if cached:
            data = json.loads(cached)
            return SQLResponse(
                sql=data["sql"],
                latency_ms=(time.perf_counter() - t_start) * 1000,
                from_cache=True,
            )

    # ── Step 2: Generate SQL ─────────────────────────────────────
    sql = await asyncio.get_event_loop().run_in_executor(
        None,
        state.model_loader.generate,
        request.question,
        request.schema,
    )

    latency_ms = (time.perf_counter() - t_start) * 1000

    # ── Step 3: Cache write ──────────────────────────────────────
    if state.cache:
        await state.cache.set(cache_key, json.dumps({"sql": sql}), ttl=3600)

    return SQLResponse(
        sql=sql,
        latency_ms=latency_ms,
        from_cache=False,
    )


@app.post("/generate-sql/stream")
async def generate_sql_stream(request: SQLRequest):
    """Streaming SQL generation — tokens arrive as they're generated."""
    if state.model_loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    async def token_generator() -> AsyncGenerator[str, None]:
        async for token in state.model_loader.generate_stream(request.question, request.schema):
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_generator(), media_type="text/event-stream")


@app.get("/cache/stats")
async def cache_stats():
    """Return Redis cache hit/miss statistics."""
    if state.cache is None:
        return {"error": "Cache not connected"}
    return await state.cache.stats()
