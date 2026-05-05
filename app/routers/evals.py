"""Eval endpoints: latest report + manual run trigger."""

import logging

import redis.asyncio as aioredis
from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.database import get_db
from app.models.eval_score import EvalScore
from app.redis_client import get_redis
from app.schemas.analytics import EvalReport
from app.services.evaluator import passes_threshold, run_evaluation

logger = logging.getLogger("paper_rag")
router = APIRouter(prefix="/evals", tags=["Evals"])
settings = get_settings()


@router.get("/report", response_model=EvalReport)
async def latest_eval(db: AsyncSession = Depends(get_db)) -> EvalReport:
    row = (
        await db.execute(select(EvalScore).order_by(EvalScore.created_at.desc()).limit(1))
    ).scalar_one_or_none()
    if row is None:
        return EvalReport(
            threshold=settings.eval_faithfulness_threshold,
            passes_threshold=False,
            notes="No eval runs yet.",
        )
    return EvalReport(
        last_run_at=row.created_at,
        faithfulness_score=row.faithfulness_score,
        relevance_score=row.relevance_score,
        citation_accuracy=row.citation_accuracy,
        num_queries_evaluated=row.num_queries_evaluated,
        threshold=settings.eval_faithfulness_threshold,
        passes_threshold=passes_threshold(
            row.faithfulness_score, settings.eval_faithfulness_threshold
        ),
        notes=row.notes,
    )


@router.post("/run", response_model=EvalReport)
async def trigger_eval(
    db: AsyncSession = Depends(get_db),
    redis: aioredis.Redis = Depends(get_redis),
) -> EvalReport:
    score = await run_evaluation(db, redis, run_label="manual")
    return EvalReport(
        last_run_at=score.created_at,
        faithfulness_score=score.faithfulness_score,
        relevance_score=score.relevance_score,
        citation_accuracy=score.citation_accuracy,
        num_queries_evaluated=score.num_queries_evaluated,
        threshold=settings.eval_faithfulness_threshold,
        passes_threshold=passes_threshold(
            score.faithfulness_score, settings.eval_faithfulness_threshold
        ),
        notes=score.notes,
    )
