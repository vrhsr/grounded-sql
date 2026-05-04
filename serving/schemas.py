"""serving/schemas.py — Pydantic request/response models"""

from pydantic import BaseModel, Field
from typing import Optional


class SQLRequest(BaseModel):
    question: str = Field(..., description="Natural language question", example="Show total revenue by category last month")
    schema: str = Field(..., description="CREATE TABLE statements for the database", example="CREATE TABLE orders (order_id INT, amount REAL, category TEXT, created_at DATETIME);")
    stream: bool = Field(default=False, description="Enable streaming token generation")
    max_new_tokens: int = Field(default=256, ge=10, le=512)
    temperature: float = Field(default=0.1, ge=0.0, le=1.0)


class SQLResponse(BaseModel):
    sql: str = Field(..., description="Generated SQL query")
    latency_ms: float = Field(..., description="End-to-end latency in milliseconds")
    from_cache: bool = Field(..., description="Whether result was served from Redis cache")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    gpu_memory: Optional[str] = None
