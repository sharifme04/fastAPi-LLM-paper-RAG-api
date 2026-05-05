"""Analytics endpoint."""

import logging

from fastapi import APIRouter, Depends
from sqlalchemy import Float, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.chunk import Chunk
from app.models.document import Document
from app.models.query import Query
from app.schemas.analytics import AnalyticsSummary, CostSummary
from app.services.cost import cost_summary

logger = logging.getLogger("paper_rag")
router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/summary", response_model=AnalyticsSummary)
async def analytics_summary(db: AsyncSession = Depends(get_db)) -> AnalyticsSummary:
    total_docs = (await db.execute(select(func.count(Document.id)))).scalar() or 0
    total_chunks = (await db.execute(select(func.count(Chunk.id)))).scalar() or 0
    total_queries = (await db.execute(select(func.count(Query.id)))).scalar() or 0

    # Average sources per answer (length of sources JSON array — Postgres jsonb_array_length).
    avg_sources = 0.0
    rows = (await db.execute(select(Query.sources))).all()
    if rows:
        lens = [len(r[0]) for r in rows if isinstance(r[0], list)]
        if lens:
            avg_sources = round(sum(lens) / len(lens), 2)

    cs = await cost_summary(db)
    return AnalyticsSummary(
        total_documents=int(total_docs),
        total_chunks=int(total_chunks),
        total_queries=int(total_queries),
        cost_summary=CostSummary(**cs),
        avg_sources_per_answer=avg_sources,
    )
