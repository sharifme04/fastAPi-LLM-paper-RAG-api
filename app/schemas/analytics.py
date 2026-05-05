"""Analytics + eval report schemas."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class CostSummary(BaseModel):
    total_cost: float
    total_queries: int
    total_cache_hits: int
    cache_hit_rate: float
    avg_cost_per_query: float


class AnalyticsSummary(BaseModel):
    total_documents: int
    total_chunks: int
    total_queries: int
    cost_summary: CostSummary
    avg_sources_per_answer: float


class EvalReport(BaseModel):
    last_run_at: Optional[datetime] = None
    faithfulness_score: Optional[float] = None
    relevance_score: Optional[float] = None
    citation_accuracy: Optional[float] = None
    num_queries_evaluated: int = 0
    threshold: float
    passes_threshold: bool = False
    notes: Optional[str] = None
