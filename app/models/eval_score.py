"""Per-run eval scores — one row per evaluation run."""

from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class EvalScore(Base):
    __tablename__ = "eval_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_label: Mapped[str] = mapped_column(String(100), nullable=False, default="manual")
    faithfulness_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    relevance_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    citation_accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    num_queries_evaluated: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    detail: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
