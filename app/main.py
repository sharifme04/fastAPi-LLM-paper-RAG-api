"""FastAPI entry point."""

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.config import get_settings
from app.database import close_db, init_db
from app.redis_client import close_redis
from app.routers import analytics, documents, evals, health, query
from app.utils.exceptions import register_exception_handlers
from app.utils.logging import generate_request_id, logger, request_id_ctx

settings = get_settings()
limiter = Limiter(key_func=get_remote_address, default_limits=[settings.rate_limit])


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("Starting application", extra={"version": settings.app_version})
    await init_db()
    logger.info("DB ready")
    yield
    await close_db()
    await close_redis()


app = FastAPI(
    title="Paper RAG API",
    description=(
        "Scientific paper Q&A: PDF ingestion → semantic chunking → "
        "sentence-transformers embeddings → pgvector retrieval → cross-encoder "
        "reranking → grounded answer with citations. Includes a small eval framework."
    ),
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.middleware("http")
async def logging_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
    rid = generate_request_id()
    request_id_ctx.set(rid)
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "Request completed",
        extra={
            "request_id": rid,
            "method": request.method,
            "path": str(request.url.path),
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        },
    )
    response.headers["X-Request-ID"] = rid
    return response


register_exception_handlers(app)

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(query.router)
app.include_router(analytics.router)
app.include_router(evals.router)


@app.get("/", tags=["Root"])
async def root() -> dict:
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "upload": "POST /documents/upload",
            "list_docs": "GET /documents",
            "ask": "POST /query",
            "analytics": "GET /analytics/summary",
            "eval_report": "GET /evals/report",
            "run_eval": "POST /evals/run",
        },
    }
