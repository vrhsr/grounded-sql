# Text-to-SQL: Fine-tuned Mistral-7B with Execution-Based Evaluation

> **One-sentence pitch:** Fine-tuned Mistral-7B on Text-to-SQL using QLoRA, ran a four-way comparison between base model, RAG-only, fine-tuned, and fine-tuned-plus-RAG evaluated by **actually executing** the generated SQL against databases — not string matching — achieving **70.2% execution accuracy** versus 53.3% baseline, deployed via FastAPI + Redis at 340ms p95 latency.

---

## Architecture

```
Natural Language Question
        │
        ▼
┌──────────────────┐    ┌─────────────────────┐
│  Redis Cache     │◄───│  FastAPI /generate  │
│  (1hr TTL)       │    │  -sql endpoint      │
└──────────────────┘    └─────────┬───────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │  Hybrid Schema Linker      │
                    │  BM25 (exact column match) │
                    │  + FAISS (semantic search) │
                    │  + RRF fusion              │
                    └─────────────┬─────────────┘
                                  │ Relevant tables only
                    ┌─────────────▼─────────────┐
                    │  INT8 Quantized Model      │
                    │  Mistral-7B + LoRA r=64   │
                    │  Fine-tuned on Spider      │
                    └─────────────┬─────────────┘
                                  │
                              SQL Output
                                  │
                    ┌─────────────▼─────────────┐
                    │  Execution Evaluator       │
                    │  Run SQL on SQLite         │
                    │  Compare result rows       │
                    └───────────────────────────┘
```

---

## Four-Way Comparison Results

| System | Execution Accuracy | Hallucinated Columns | p95 Latency |
|---|---|---|---|
| A: Base Mistral-7B | 53.3% | ~30% of errors | 5.1s |
| B: RAG Only | 54.2% | ~18% of errors | 4.3s |
| C: Fine-tuned (r=64) | **70.2%** | **0% of errors** | 5.8s |
| D: Fine-tuned + RAG | 66.4% | **0% of errors** | 7.1s |

> **Non-obvious finding:** On truly novel schemas (166 held-out databases), RAG-only barely beats the base model, and cross-schema RAG examples actually hurt the fine-tuned model (70.2% → 66.4%). Retrieval quality is the ceiling for generalization, not model capacity.

---

## Quantization Comparison

| Format | Model Size | Throughput | p95 Latency | Accuracy Drop |
|---|---|---|---|---|
| fp16 | 14GB | 18 tok/s | 1.8s | baseline |
| INT8 | 7GB | 41 tok/s | 0.8s | -0.8% |
| GGUF Q4_K_M | 4GB | 67 tok/s | 0.5s | -3.1% |

> **Production choice:** INT8 — 2.3x throughput with <1% accuracy drop. Q4 matters for chatbots; for SQL generation where correctness is binary, the 3.1% drop is too costly.

---

## LoRA Rank Ablation

| Run | Rank | Behavior |
|---|---|---|
| A | r=8 | Early plateau — underfitting, can't adapt to complex schemas |
| B | r=16 | Steady decrease then stabilization — execution accuracy 66.2% |
| C | r=64 | Higher val loss but better generation — execution accuracy **70.2%** (Optimal) |

---

## Error Taxonomy (System C: Fine-tuned Model)

| Error Category | % of Errors |
|---|---|
| Correct SQL, wrong result | 36.0% |
| Syntax error | 31.8% (down 35% after SQL post-processing) |
| Wrong JOIN logic | 26.6% |
| Wrong aggregation | 5.5% |
| Wrong WHERE / filters | 0.0% |
| Hallucinated column | **0.0%** (eliminated by hybrid schema linking) |

---

## Project Structure

```
text-to-sql/
├── data_pipeline.py          ← System 1: Spider → instruction format
├── training/
│   ├── config.yaml           ← Single source of truth (all hyperparams)
│   ├── train.py              ← QLoRA fine-tuning (4-bit + LoRA)
│   ├── dataset.py            ← HF Dataset with label masking
│   └── ablations/            ← rank_8.yaml, rank_16 (default), rank_64.yaml
├── evaluation/
│   ├── executor.py           ← Binary SQL execution evaluator (core)
│   ├── four_way_compare.py   ← Runs all 4 systems on 1034 test queries
│   ├── error_taxonomy.py     ← Categorizes failure modes
│   └── tanglish_eval.py      ← Tanglish ablation (Zoho/Freshworks card)
├── retrieval/
│   ├── schema_index.py       ← FAISS dense index
│   ├── bm25_index.py         ← BM25 sparse index (exact column match)
│   └── hybrid_linker.py      ← RRF fusion — attacks hallucinated columns
├── serving/
│   ├── main.py               ← FastAPI app with streaming
│   ├── model_loader.py       ← INT8 model inference
│   ├── cache.py              ← Redis async cache
│   └── schemas.py            ← Pydantic models
├── data/
│   ├── processed/            ← train.jsonl, validation.jsonl, test.jsonl
│   └── raw/synthetic/        ← e-commerce + Tanglish data
├── dataset/spider/           ← Original Spider dataset (already present)
├── docker-compose.yml        ← Redis + API
├── Dockerfile
└── requirements.txt
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run data pipeline (already done — outputs in data/processed/)
```bash
python data_pipeline.py --config training/config.yaml
```

### 3. Build retrieval indexes
```bash
python retrieval/hybrid_linker.py --build \
    --tables dataset/spider/tables.json \
    --train data/processed/train.jsonl
```

### 4. Train (rank-64, default)
```bash
python training/train.py --config training/ablations/rank_64_v2.yaml
```

### 5. Ablation runs
```bash
python training/train.py --config training/config.yaml --override training/ablations/rank_8.yaml
python training/train.py --config training/config.yaml
```

### 6. Four-way evaluation
```bash
# Baseline only (no GPU needed)
python evaluation/four_way_compare.py --systems A

# Full comparison (requires fine-tuned adapter)
python evaluation/four_way_compare.py \
    --finetuned-adapter checkpoints/rank_64/final_adapter \
    --systems A B C D
```

### 7. Error analysis
```bash
python evaluation/error_taxonomy.py \
    --results evaluation/results/c:_fine-tuned_results.csv
```

### 8. Tanglish evaluation
```bash
python evaluation/tanglish_eval.py \
    --adapter checkpoints/rank_64/final_adapter
```

### 9. Serve the API
```bash
# With Docker (Redis + API)
docker-compose up

# Or directly
MODEL_ADAPTER_PATH=checkpoints/rank_64/final_adapter \
uvicorn serving.main:app --host 0.0.0.0 --port 8000
```

---

## Dataset

- **Spider** (primary): 7,000 train + 1,659 others + 1,034 dev/test
- **Processed**: 8,080 train | 495 validation | 1,034 test
- **Filtered**: 84 sequences (>400 words) — mostly 8+ table schemas
- **Schema**: 166 databases across 138 domains

---

## Key Design Decisions

1. **Execution accuracy over string match** — Two SQLs can look completely different but return identical rows. Only execution tells you if a query is correct.

2. **BM25 for exact column names** — Semantic search finds "revenue" → `amount`, but BM25 finds `customer_id` exactly. Both are needed.

3. **INT8 over Q4 for SQL** — SQL correctness is binary. A 3.1% accuracy drop from Q4 quantization means ~32 more wrong queries on the 1034 test set.

4. **Label masking on prompt** — Training loss is computed only on SQL tokens, not on the schema/question prefix. This ensures the model learns to generate SQL, not to predict the schema.

5. **Tanglish as a documented failure** — Not a weakness to hide. It's a research finding: the model breaks on Tamil-English code-switched queries. Identifying it precisely (schema linking failure on Romanized Tamil words) is more valuable than pretending it doesn't exist.
