# প্রজেক্ট ৩ — Paper RAG API (বাংলা ফ্লো ডকুমেন্টেশন)

এই ডকুমেন্টে পুরো প্রজেক্টটা **কোন ফাইল কী কাজ করে**, **PDF থেকে answer পর্যন্ত পথ**, এবং **eval framework কীভাবে ডিপ্লয় গেট করে** — সব **বাংলায়** আছে।

---

## ১. প্রজেক্ট কী?

ব্যবহারকারী scientific paper (PDF) আপলোড করে, তারপর প্রশ্ন করে। সিস্টেম:
1. PDF থেকে text extract করে (pymupdf দিয়ে)
2. semantic chunking — sentence-aware, ৫১২ token target, ১০০ token overlap
3. sentence-transformers এ embed করে (3৮৪ dim)
4. PostgreSQL + **pgvector** এ store করে
5. প্রশ্ন আসলে → query embed → pgvector cosine search → top 10 → cross-encoder rerank → top 5
6. Claude-কে retrieved passages + question পাঠিয়ে structured JSON answer + citations [1] [2] পায়
7. সব Redis-এ cache হয় (১ ঘণ্টা TTL)

এছাড়া আছে **eval framework** — golden dataset দিয়ে faithfulness, relevance, citation accuracy স্কোর। CI-তে threshold পেরোলে deploy block।

---

## ২. ডিরেক্টরি স্ট্রাকচার

```
project-3-paper-rag-api/
├── app/
│   ├── main.py                FastAPI অ্যাপ এন্ট্রি
│   ├── config.py              .env থেকে settings
│   ├── database.py            async SQLAlchemy + pgvector extension setup
│   ├── redis_client.py        Redis pool
│   ├── models/                ORM মডেল
│   │   ├── _vector_compat.py  ★ cross-dialect Vector column (Postgres = pgvector, SQLite = JSON)
│   │   ├── document.py        upload-করা PDF
│   │   ├── chunk.py           chunk text + embedding + page range
│   │   ├── query.py           question + answer + sources + cost
│   │   ├── feedback.py        helpful/not feedback
│   │   └── eval_score.py      eval score per run
│   ├── schemas/               Pydantic schema
│   ├── services/              ★ মূল business logic
│   │   ├── pdf_parser.py      pymupdf দিয়ে page-by-page text extract
│   │   ├── chunker.py         sentence boundary + overlap-aware chunking
│   │   ├── embedder.py        sentence-transformers singleton (test hook সহ)
│   │   ├── ingestion.py       PDF → chunks → embeddings → DB
│   │   ├── retriever.py       pgvector cosine search + cross-encoder rerank
│   │   ├── generator.py       Claude API call structured JSON output সহ
│   │   ├── qa.py              ★★ end-to-end orchestration: cache → retrieve → generate → persist
│   │   ├── cost.py            token → USD, daily aggregate
│   │   └── evaluator.py       LLM-as-judge eval (faithfulness/relevance/citation)
│   ├── routers/
│   │   ├── health.py          /health
│   │   ├── documents.py       upload / list / delete
│   │   ├── query.py           ★ POST /query (মূল endpoint)
│   │   ├── analytics.py       per-doc summary
│   │   └── evals.py           eval report + manual run
│   └── utils/
│       ├── logging.py         JSON log with request_id
│       └── exceptions.py      AppError + handlers
├── eval_dataset/golden.json   eval-এর জন্য golden questions
├── tests/                     19 টা টেস্ট, সবগুলো pass
├── requirements.txt
├── Dockerfile                 HF মডেল pre-cache সহ
├── docker-compose.yml         api + pgvector/pgvector:pg16 + redis:7
└── .env.example
```

---

## ৩. প্রতিটা ফাইলের কাজ এক লাইনে

### Entry layer

| ফাইল | কাজ |
|------|-----|
| [app/main.py](app/main.py) | FastAPI app, middleware (CORS, rate limit, request log), router mount, lifespan-এ DB তৈরি |
| [app/config.py](app/config.py) | `.env` থেকে DATABASE_URL, REDIS_URL, ANTHROPIC_API_KEY, embedding model, retrieval params সব load |
| [app/database.py](app/database.py) | async engine + session factory + **`CREATE EXTENSION IF NOT EXISTS vector`** PostgreSQL-এ |
| [app/redis_client.py](app/redis_client.py) | async Redis pool |

### Models

| ফাইল | কী রাখে |
|------|---------|
| [app/models/_vector_compat.py](app/models/_vector_compat.py) | ★ smart `VectorCompat` column — Postgres-এ `pgvector.Vector(384)`, SQLite-এ JSON fallback (test-এর জন্য) |
| [app/models/document.py](app/models/document.py) | uploaded PDF — filename, file_path, total_chunks, text_length |
| [app/models/chunk.py](app/models/chunk.py) | এক row per chunk — text, embedding, page_start, page_end, token_count |
| [app/models/query.py](app/models/query.py) | প্রতিটা প্রশ্ন — question, answer, sources (JSON), retrieved_chunk_ids, tokens, cost, cache_hit |
| [app/models/feedback.py](app/models/feedback.py) | helpful/not feedback (eval tracking-এর জন্য) |
| [app/models/eval_score.py](app/models/eval_score.py) | eval run-এর পর per-run score |

### Services

| ফাইল | কাজ |
|------|-----|
| [app/services/pdf_parser.py](app/services/pdf_parser.py) | pymupdf (`fitz`) দিয়ে PDF খোলে, প্রতিটা page-এর text return করে। OCR দরকার হলে error দেয় |
| [app/services/chunker.py](app/services/chunker.py) | sentences কে accumulate করে ~512 token হলে chunk emit, পরের chunk ১০০ token overlap রাখে |
| [app/services/embedder.py](app/services/embedder.py) | sentence-transformers singleton (lazy load), `set_embedder()` test hook |
| [app/services/ingestion.py](app/services/ingestion.py) | PDF bytes → parse → chunk → embed (batch) → DB save |
| [app/services/retriever.py](app/services/retriever.py) | ★ Postgres-এ pgvector `<=>` cosine ANN, SQLite-এ Python cosine fallback; cross-encoder rerank |
| [app/services/generator.py](app/services/generator.py) | retrieved passages + question → Claude → strict JSON answer parse |
| [app/services/qa.py](app/services/qa.py) | ★★ মূল orchestration: Redis cache check → cost check → retrieve → generate → DB save → cache write |
| [app/services/cost.py](app/services/cost.py) | per-call USD হিসাব, today's total, limit গেট |
| [app/services/evaluator.py](app/services/evaluator.py) | LLM-as-judge — golden dataset-এর প্রতিটা question-এ qa চালায়, Claude-কে judge হিসেবে use করে |

### Routers

| ফাইল | endpoint |
|------|----------|
| [app/routers/health.py](app/routers/health.py) | `GET /health` (DB + Redis + pgvector extension status) |
| [app/routers/documents.py](app/routers/documents.py) | `POST /documents/upload`, `GET /documents`, `GET/DELETE /documents/{id}` |
| [app/routers/query.py](app/routers/query.py) | ★ `POST /query`, `GET /query/{id}`, `POST /query/{id}/feedback` |
| [app/routers/analytics.py](app/routers/analytics.py) | `GET /analytics/summary` |
| [app/routers/evals.py](app/routers/evals.py) | `GET /evals/report`, `POST /evals/run` |

---

## ৪. পুরো PDF → answer পথ

```
[ব্যবহারকারী] POST /documents/upload (PDF)
    │
    ▼
[routers/documents.py]
    │
    ▼
[services/ingestion.py: ingest_pdf()]
    ├── pdf_parser.extract_pdf_pages()  (pymupdf দিয়ে page-wise text)
    ├── chunker.chunk_pages()            (sentence-aware chunk + overlap)
    ├── embedder.embed_texts()           (sentence-transformers batch)
    └── DB save: Document + Chunks (each with embedding)


[ব্যবহারকারী] POST /query {"question": "..."}
    │
    ▼
[routers/query.py]
    │
    ▼
[services/qa.py: answer_question()]
    │
    ├── 1. Redis cache check (sha256(question) key, 1h TTL)
    │       └── hit → return cached, no API call, cost=0, cache_hit=true
    │
    ├── 2. cost.check_cost_limit()  (daily limit hit হলে 429)
    │
    ├── 3. retriever.retrieve()
    │       ├── embedder.embed_query()  (একই model use করে)
    │       ├── pgvector cosine search → top 10 candidates
    │       └── cross-encoder rerank → top 5
    │
    ├── 4. generator.generate_answer()
    │       ├── numbered passages-এ inject
    │       ├── Claude → strict JSON {answer, used_passages}
    │       └── parse + validate
    │
    ├── 5. cost.calculate_cost()
    │
    ├── 6. DB-তে Query row save (sources, chunk_ids, tokens, cost সহ)
    │
    └── 7. Redis-এ cache write
    │
    ▼
Response: {answer, sources[], cost, cache_hit}
```

---

## ৫. pgvector কীভাবে use করা হয়?

[app/services/retriever.py](app/services/retriever.py)-এ:

```python
# Postgres branch — pgvector cosine distance operator
sql = """
  SELECT c.id, c.text, ..., 1 - (c.embedding <=> '[...]'::vector) AS similarity
  FROM chunks c JOIN documents d ON d.id = c.document_id
  WHERE c.embedding IS NOT NULL
  ORDER BY c.embedding <=> '[...]'::vector ASC
  LIMIT 10
"""
```

`<=>` হলো pgvector-এর cosine distance operator। `1 - distance = similarity`। এটা ANN-style fast lookup।

SQLite test-এ আমরা সব chunk pull করে Python-এ cosine হিসাব করি — slow, কিন্তু test-এর জন্য fine।

`models/_vector_compat.py`-এর `VectorCompat` column type এই dialect switch automatic handle করে।

---

## ৬. Cross-encoder reranking কেন?

Vector search semantically similar chunk দেয়, কিন্তু always relevant না। Cross-encoder (`ms-marco-MiniLM-L-6-v2`) প্রতিটা (query, chunk) pair-এ একসাথে দেখে — slow but more accurate।

Strategy:
1. Vector search → top 10 (cheap, fast)
2. Cross-encoder → reorder, take top 5 (slow, accurate)

এটা production RAG-এর standard pattern।

---

## ৭. Eval framework

[app/services/evaluator.py](app/services/evaluator.py)-এ light-weight LLM-as-judge:

- `eval_dataset/golden.json` — `[{question, expected_answer, expected_chunk_keywords[]}]`
- প্রতিটা question-এ `answer_question()` চালাই
- তারপর Claude-কে judge হিসেবে call করি — তিনটা score দেয় 0..1:
  - **faithfulness** — answer-এর claims কি retrieved context-এ আছে?
  - **relevance** — answer কি question-এর জবাব দিচ্ছে?
  - **citation_accuracy** — `[n]` citations কি actually সেই passage-কে support করে?
- average score persist → `EvalScore` table-এ
- threshold (default 0.7) পেরোলে `passes_threshold=true`

CI-তে use করার pattern:
```yaml
- run: pytest
- run: |
    curl -fsS -X POST http://localhost:8000/evals/run | tee eval.json
    python -c "import json,sys; r=json.load(open('eval.json'));
               sys.exit(0 if r['passes_threshold'] else 1)"
```

API key না থাকলে heuristic word-overlap scorer use হয় — তাই offline-ও test pass করে।

---

## ৮. Test strategy

```bash
pytest
```

মোট ১৯টা test, সবগুলো pass, ~3 second-এ চলে। কেমনে?

- **FakeEmbedder** — text-এর SHA-256 hash থেকে deterministic 384-dim vector। same text → same vector।
- **FakeReranker** — query/candidate Jaccard token overlap।
- **SQLite + aiosqlite** — pgvector নেই, retriever-এর Python cosine fallback use হয়।
- **FakeRedis** — in-memory dict।
- **Anthropic patch** — `make_anthropic_response()` দিয়ে structured JSON answer mock।

ফলে কোনো network/torch download/Postgres লাগে না।

---

## ৯. Docker compose-এ কী কী চলে

```yaml
services:
  api:        FastAPI, port 8000
  postgres:   pgvector/pgvector:pg16  ← pgvector pre-installed
  redis:      redis:7-alpine
```

```bash
cp .env.example .env  # ANTHROPIC_API_KEY বসাও
docker compose up -d
curl http://localhost:8000/health
# { db: connected, redis: connected, pgvector: installed }
```

> ⚠️ **First build** ~5-10 মিনিট নেয় — sentence-transformers + PyTorch ~2GB download। Dockerfile-এ HF model pre-cache করা আছে যাতে container start fast হয়।

---

## ১০. একনজরে — কোন ফাইল কোন কাজের জন্য

| কাজ | ফাইল |
|-----|------|
| FastAPI + lifespan + middleware | [app/main.py](app/main.py) |
| `.env` config | [app/config.py](app/config.py) |
| pgvector extension setup | [app/database.py](app/database.py) |
| Cross-dialect Vector column | [app/models/_vector_compat.py](app/models/_vector_compat.py) |
| PDF parsing | [app/services/pdf_parser.py](app/services/pdf_parser.py) |
| Sentence-aware chunking | [app/services/chunker.py](app/services/chunker.py) |
| Embedder + reranker singleton | [app/services/embedder.py](app/services/embedder.py) |
| **Vector search + rerank** | [app/services/retriever.py](app/services/retriever.py) |
| **Claude answer with citations** | [app/services/generator.py](app/services/generator.py) |
| **Cache → retrieve → generate orchestrator** | [app/services/qa.py](app/services/qa.py) |
| Cost tracking | [app/services/cost.py](app/services/cost.py) |
| Eval framework (LLM-as-judge) | [app/services/evaluator.py](app/services/evaluator.py) |
| Routers | [app/routers/](app/routers/) |
| Tests | [tests/](tests/) |

---

**সারসংক্ষেপ:** PDF → `ingestion.py` → DB; Question → `qa.py` → cache check → `retriever.py` (pgvector + cross-encoder) → `generator.py` (Claude JSON) → DB + cache → response।
