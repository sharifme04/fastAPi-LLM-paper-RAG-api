"""Query endpoint: ask a question, get a grounded answer with citations."""

import logging

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.feedback import Feedback
from app.models.query import Query
from app.redis_client import get_redis
from app.schemas.query import (
    AnswerResponse,
    FeedbackRequest,
    QuestionRequest,
    Source,
)
from app.services.qa import answer_question

logger = logging.getLogger("paper_rag")
router = APIRouter(tags=["Query"])


def _to_response(q: Query) -> AnswerResponse:
    sources = [Source(**s) for s in (q.sources or [])]
    return AnswerResponse(
        query_id=q.id,
        question=q.question,
        answer=q.answer or "",
        sources=sources,
        cache_hit=q.cache_hit,
        tokens_used=q.input_tokens + q.output_tokens,
        cost=q.cost,
        created_at=q.created_at,
    )


@router.post("/query", response_model=AnswerResponse)
async def post_query(
    payload: QuestionRequest,
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
) -> AnswerResponse:
    q = await answer_question(
        db, redis, question=payload.question, top_k_rerank=payload.top_k
    )
    return _to_response(q)


@router.get("/query/{query_id}", response_model=AnswerResponse)
async def get_query(
    query_id: int,
    db: AsyncSession = Depends(get_db),
) -> AnswerResponse:
    q = (
        await db.execute(select(Query).where(Query.id == query_id))
    ).scalar_one_or_none()
    if q is None:
        raise HTTPException(status_code=404, detail=f"Query {query_id} not found")
    return _to_response(q)


@router.post("/query/{query_id}/feedback", status_code=201)
async def submit_feedback(
    query_id: int,
    payload: FeedbackRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    q = (
        await db.execute(select(Query).where(Query.id == query_id))
    ).scalar_one_or_none()
    if q is None:
        raise HTTPException(status_code=404, detail=f"Query {query_id} not found")
    fb = Feedback(query_id=query_id, helpful=payload.helpful, notes=payload.notes)
    db.add(fb)
    await db.flush()
    return {"feedback_id": fb.id, "query_id": query_id, "helpful": fb.helpful}
