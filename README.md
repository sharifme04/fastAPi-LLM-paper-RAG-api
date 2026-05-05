# Paper RAG API

Scientific paper Q&A — **upload PDFs**, ask questions, get **grounded answers with citations**. End-to-end RAG pipeline with **pgvector** vector storage, **cross-encoder reranking**, **Redis** query caching, and a **lightweight LLM-as-judge eval framework** (faithfulness / relevance / citation accuracy) gated at a configurable threshold.

This is **Project 3** of the 5-project AI engineering portfolio. It's the project the original learning plan calls "the recommended portfolio piece" because RAG appears in 80%+ of AI engineering postings.

---

## Pipeline at a glance

```
PDF upload
   │
   ▼
[pymupdf] page-by-page text extraction
   │
   ▼
[chunker]  semantic chunking with ~512 tok target, 100 tok overlap
   │
   ▼
[sentence-transformers/all-MiniLM-L6-v2]  384-dim embeddings (batched)
   │
   ▼
PostgreSQL + pgvector  (chunks.embedding column)


Query flow:
   user question
   │
   ▼  Redis cache check (1 h TTL on the question)
   │
   ▼  embed query (same model)
   │
   ▼  pgvector cosine search → top 10 candidates
   │
   ▼  cross-encoder/ms-marco-MiniLM-L-6-v2 → rerank → top 5
   │
   ▼  Claude with structured-JSON system prompt → answer + cited passage indices
   │
   ▼  persist Query row + sources, write to Redis cache
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI + Uvicorn |
| Database | PostgreSQL + pgvector extension + asyncpg |
| Vector DB | pgvector (native to PostgreSQL) |
| Cache | Redis (query result caching) |
| ORM | SQLAlchemy 2.0 (async) + custom Vector column |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 (384-dim, fast, accurate) |
| Re-ranking | sentence-transformers cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Document Parsing | pymupdf (fitz) — robust PDF text extraction |
| LLM | Anthropic Claude Sonnet |
| Validation | Pydantic v2 + pydantic-settings |
| Evaluation | ragas (Retrieval-Augmented Generation Evaluation) |
| Rate Limiting | slowapi |
| Testing | pytest + pytest-asyncio |
| Logging | python-json-logger (structured JSON) |
| Containerisation | Docker Compose |

## Features

- ✅ **PDF Upload & Ingestion** — Parse PDFs, extract text, validate content automatically
- ✅ **Semantic Chunking** — Split on sentence boundaries with configurable overlap (512 tokens, 100-token overlap)
- ✅ **Vector Embeddings** — Generate and store 384-dim embeddings in pgvector
- ✅ **Similarity Search** — pgvector cosine similarity → top-10 candidates
- ✅ **Cross-Encoder Re-ranking** — Rerank candidates with MS MARCO cross-encoder → top-5 quality chunks
- ✅ **Grounded Answer Generation** — Claude generates answers with structured citations (document, page, chunk_id)
- ✅ **Redis Query Caching** — Cache full answers for 1 hour (hash of question as key)
- ✅ **Cost Tracking** — Per-query token usage, daily budget limits, cost alerts
- ✅ **Evaluation Framework** — ragas metrics (faithfulness, relevance, citation accuracy) gated in CI
- ✅ **Analytics Dashboard** — Per-document stats, cost summary, cache hit rate
- ✅ **Structured Logging** — JSON logs with request_id, retrieval scores
- ✅ **Health Checks** — Verify DB + pgvector extension + Redis connectivity
- ✅ **Feedback Collection** — Users mark answers as helpful/incorrect for eval improvement

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Health check (DB + pgvector + Redis status) |
| POST | `/documents/upload` | Upload a PDF document |
| GET | `/documents` | List all uploaded documents |
| GET | `/documents/{id}` | Get document metadata + chunk count |
| DELETE | `/documents/{id}` | Delete document (cascade deletes chunks) |
| POST | `/query` | Ask a question, get grounded answer with citations |
| GET | `/query/{id}` | Retrieve past query + answer + sources |
| POST | `/query/{id}/feedback` | Mark answer as helpful or incorrect |
| GET | `/analytics/summary` | Per-document + cost summary |
| GET | `/evals/report` | Eval scores (faithfulness, relevance, citation) |
| POST | `/evals/run` | Trigger eval suite on golden dataset |

---

```
project-3-paper-rag-api/
├── app/
│   ├── main.py                    FastAPI entry — lifespan creates pgvector ext + tables
│   ├── config.py                  pydantic-settings (.env)
│   ├── database.py                async SQLAlchemy + pgvector ext setup
│   ├── redis_client.py            async Redis pool
│   ├── models/
│   │   ├── _vector_compat.py      cross-dialect Vector column (pgvector ↔ JSON SQLite fallback)
│   │   ├── document.py            uploaded PDFs
│   │   ├── chunk.py               chunk text + embedding + page range
│   │   ├── query.py               question + answer + sources + cost
│   │   ├── feedback.py            helpful/not on past answers
│   │   └── eval_score.py          per-run eval scores
│   ├── schemas/
│   ├── services/
│   │   ├── pdf_parser.py          pymupdf wrapper
│   │   ├── chunker.py             sentence-aware chunking with overlap
│   │   ├── embedder.py            sentence-transformers singleton + test hooks
│   │   ├── ingestion.py           orchestrates PDF → chunks → embeddings → DB
│   │   ├── retriever.py           pgvector cosine + cross-encoder rerank (SQLite fallback)
│   │   ├── generator.py           Claude with structured JSON output + citations
│   │   ├── qa.py                  end-to-end: cache → retrieve → generate → persist
│   │   ├── cost.py                token → USD, daily aggregate, limit gate
│   │   └── evaluator.py           LLM-as-judge eval (faithfulness/relevance/citation)
│   ├── routers/
│   │   ├── health.py              /health (DB + Redis + pgvector status)
│   │   ├── documents.py           upload / list / get / delete
│   │   ├── query.py               POST /query, GET /query/{id}, /feedback
│   │   ├── analytics.py           per-doc + cost summary
│   │   └── evals.py               GET /evals/report, POST /evals/run
│   └── utils/
│       ├── logging.py             structured JSON logs w/ request_id
│       └── exceptions.py          AppError + global FastAPI handlers
├── eval_dataset/golden.json       seed eval questions
├── tests/                         19 passing (mocks the embedder + Anthropic SDK)
├── requirements.txt
├── pyproject.toml
├── Dockerfile                     pre-caches the HF models in the image
├── docker-compose.yml             api + pgvector/pgvector:pg16 + redis:7
└── .env.example
```

---

## API

| Method | Path                              | Purpose                                       |
|--------|-----------------------------------|-----------------------------------------------|
| GET    | `/health`                         | DB + Redis + pgvector extension status        |
| POST   | `/documents/upload`               | Upload a PDF (multipart)                      |
| GET    | `/documents`                      | List all uploaded documents                   |
| GET    | `/documents/{id}`                 | Document detail                               |
| DELETE | `/documents/{id}`                 | Delete document + cascade chunks              |
| POST   | `/query`                          | Ask a question, get a grounded answer         |
| GET    | `/query/{id}`                     | Recall a past query                           |
| POST   | `/query/{id}/feedback`            | Mark answer helpful / not                     |
| GET    | `/analytics/summary`              | Per-doc + cost summary                        |
| GET    | `/evals/report`                   | Latest eval scores + threshold pass/fail      |
| POST   | `/evals/run`                      | Trigger an eval run on the golden dataset     |
| GET    | `/docs`                           | Swagger UI                                    |

### Query example

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does attention work in transformers?"}'
```

Response:
```json
{
  "query_id": 12,
  "question": "How does attention work in transformers?",
  "answer": "Scaled dot-product attention computes weights from query and key vectors and applies them to value vectors [1]. The result lets each token weigh information from every other token in the sequence [2].",
  "sources": [
    {"document_id": 3, "document_filename": "attention.pdf", "chunk_id": 42,
     "chunk_index": 3, "page_start": 4, "page_end": 4, "relevance_score": 0.91,
     "text_preview": "Scaled dot-product attention…"}
  ],
  "cache_hit": false,
  "tokens_used": 1320,
  "cost": 0.0089,
  "created_at": "2026-05-04T18:25:00Z"
}
```

---

## Run it locally

```bash
cp .env.example .env
# set ANTHROPIC_API_KEY=sk-ant-...

docker compose up -d
curl http://localhost:8000/health
# {"status":"ok","db":"connected","redis":"connected","pgvector":"installed",...}
```

Upload a paper, then ask:
```bash
curl -F "file=@my_paper.pdf" http://localhost:8000/documents/upload
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main contribution?"}'
```

> **First request** is slow — sentence-transformers loads ~90 MB into memory the first time `get_embedder()` runs, plus the cross-encoder. Subsequent requests are fast. The Dockerfile pre-caches both models so cold container boots are quicker.

### Without Docker

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# bring up your own Postgres (with pgvector) + Redis, then:
export DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/paper_rag
export REDIS_URL=redis://localhost:6379/0
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn app.main:app --reload
```

---

## Tests

```bash
pip install -r requirements.txt
pytest
```

The test suite mocks the heavy ML deps:
- `FakeEmbedder` returns deterministic hash-bucket vectors → same text always embeds the same way.
- `FakeReranker` scores by Jaccard token overlap.
- The retriever has a SQLite-aware fallback that does cosine in Python, so **no pgvector needed for tests**.
- Anthropic SDK is patched per-test with a `make_anthropic_response` helper.

**19 / 19 passing in ~3 seconds.** Coverage:
- chunker (sentence split, overlap, page-range tracking)
- embedder + reranker (fake-injection, deterministic output)
- retriever (top-k, empty DB, SQLite cosine path)
- qa orchestrator (full flow, cache hit, cost calc)
- routes (list/get/upload/query/analytics/eval report)
- health

---

## Eval framework

Lightweight LLM-as-judge implementation in [app/services/evaluator.py](app/services/evaluator.py) — chosen instead of `ragas` to keep dependencies lean and failure modes legible.

Three metrics scored 0..1:
- **faithfulness** — are claims in the answer supported by the retrieved context?
- **relevance** — does the answer address the question?
- **citation accuracy** — do `[n]` markers point to passages that contain the cited claim?

Configure the gate via `EVAL_FAITHFULNESS_THRESHOLD` (default 0.7). The CI pattern is:

```yaml
# .github/workflows/ci.yml (excerpt)
- run: pytest
- run: |
    curl -fsS -X POST http://localhost:8000/evals/run | tee eval.json
    python -c "import json,sys; r=json.load(open('eval.json'));
               sys.exit(0 if r['passes_threshold'] else 1)"
```

Without an API key, the evaluator falls back to a heuristic word-overlap scorer so `pytest` works offline.

---

## Cross-cutting concerns

| Concern              | Implementation                                                                |
|----------------------|-------------------------------------------------------------------------------|
| Secrets              | `.env` + `pydantic-settings`, never hardcoded                                 |
| Health check         | `GET /health` — DB + Redis + pgvector extension                               |
| Structured logging   | `python-json-logger`, every request gets a `request_id` in headers + logs     |
| Rate limiting        | `slowapi`, configurable via `RATE_LIMIT`                                      |
| Error handling       | Global handlers in [app/utils/exceptions.py](app/utils/exceptions.py)         |
| Caching              | Redis SHA-256 of normalized question, 1 h TTL                                 |
| Cost control         | Per-call cost in [app/services/cost.py](app/services/cost.py), daily cap, 429 on breach |
| Container parity     | docker compose mirrors prod                                                   |

# fastAPi-LLM-paper-RAG-api
