"""End-to-end Q&A orchestration: cache → retrieve → generate → persist."""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Optional

import redis.asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.query import Query
from app.services.cost import calculate_cost, check_cost_limit
from app.services.generator import generate_answer
from app.services.retriever import Retrieved, retrieve

logger = logging.getLogger("paper_rag")
settings = get_settings()

CACHE_TTL = 3600


def _cache_key(question: str) -> str:
    h = hashlib.sha256(question.strip().lower().encode()).hexdigest()[:16]
    return f"qa:{h}"


def _build_sources(retrieved: list[Retrieved]) -> list[dict]:
    return [
        {
            "document_id": r.document_id,
            "document_filename": r.document_filename,
            "chunk_id": r.chunk_id,
            "chunk_index": r.chunk_index,
            "page_start": r.page_start,
            "page_end": r.page_end,
            "relevance_score": round(r.relevance_score, 4),
            "text_preview": r.text[:300] + ("…" if len(r.text) > 300 else ""),
        }
        for r in retrieved
    ]


async def answer_question(
    db: AsyncSession,
    redis: aioredis.Redis,
    question: str,
    top_k_rerank: Optional[int] = None,
    document_ids: Optional[list[int]] = None,
) -> Query:
    """Answer a question using the RAG pipeline. Returns the persisted Query row."""
    key = _cache_key(question)

    cached_raw = await redis.get(key)
    if cached_raw:
        cached = json.loads(cached_raw)
        logger.info("QA cache hit", extra={"question_len": len(question)})
        q = Query(
            question=question,
            answer=cached["answer"],
            sources=cached["sources"],
            retrieved_chunk_ids=cached.get("chunk_ids", []),
            input_tokens=0,
            output_tokens=0,
            cost=0.0,
            cache_hit=True,
        )
        db.add(q)
        await db.flush()
        return q

    await check_cost_limit(db)

    retrieved = await retrieve(
        db,
        question=question,
        top_k_rerank=top_k_rerank,
        document_ids=document_ids,
    )

    gen = await generate_answer(question, retrieved)
    cost = calculate_cost(gen["input_tokens"], gen["output_tokens"])

    sources = _build_sources(retrieved)
    chunk_ids = [r.chunk_id for r in retrieved]

    q = Query(
        question=question,
        answer=gen["answer"],
        sources=sources,
        retrieved_chunk_ids=chunk_ids,
        input_tokens=gen["input_tokens"],
        output_tokens=gen["output_tokens"],
        cost=cost,
        cache_hit=False,
    )
    db.add(q)
    await db.flush()

    await redis.set(
        key,
        json.dumps({"answer": gen["answer"], "sources": sources, "chunk_ids": chunk_ids}),
        ex=CACHE_TTL,
    )

    logger.info(
        "QA completed",
        extra={
            "query_id": q.id,
            "tokens": gen["input_tokens"] + gen["output_tokens"],
            "cost": cost,
            "sources": len(sources),
        },
    )
    return q
