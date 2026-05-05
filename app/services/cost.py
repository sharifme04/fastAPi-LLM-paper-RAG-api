"""Cost tracking — same shape as Project 2, applied to query rows."""

import logging
from datetime import date, datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.models.query import Query
from app.utils.exceptions import CostLimitExceededError

logger = logging.getLogger("paper_rag")
settings = get_settings()


def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    in_cost = (input_tokens / 1_000_000) * settings.anthropic_input_price_per_mtok
    out_cost = (output_tokens / 1_000_000) * settings.anthropic_output_price_per_mtok
    return round(in_cost + out_cost, 6)


async def daily_cost_total(db: AsyncSession) -> float:
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    result = await db.execute(
        select(func.coalesce(func.sum(Query.cost), 0.0)).where(Query.created_at >= today)
    )
    return float(result.scalar() or 0.0)


async def check_cost_limit(db: AsyncSession) -> None:
    total = await daily_cost_total(db)
    if total >= settings.daily_cost_limit:
        raise CostLimitExceededError(daily_total=total, limit=settings.daily_cost_limit)


async def cost_summary(db: AsyncSession) -> dict:
    """Aggregate cost summary across the queries table."""
    total_q = (await db.execute(select(func.count(Query.id)))).scalar() or 0
    total_cost = (await db.execute(select(func.coalesce(func.sum(Query.cost), 0.0)))).scalar() or 0.0
    cache_hits = (
        await db.execute(select(func.count(Query.id)).where(Query.cache_hit == True))  # noqa: E712
    ).scalar() or 0
    rate = (cache_hits / total_q * 100) if total_q else 0.0
    avg = (float(total_cost) / total_q) if total_q else 0.0
    return {
        "total_cost": round(float(total_cost), 4),
        "total_queries": int(total_q),
        "total_cache_hits": int(cache_hits),
        "cache_hit_rate": round(rate, 2),
        "avg_cost_per_query": round(avg, 6),
    }
