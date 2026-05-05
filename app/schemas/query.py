"""Pydantic schemas for the query endpoint."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class Source(BaseModel):
    document_id: int
    document_filename: str
    chunk_id: int
    chunk_index: int
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    relevance_score: float
    text_preview: str


class AnswerResponse(BaseModel):
    query_id: int
    question: str
    answer: str
    sources: list[Source]
    cache_hit: bool
    tokens_used: int
    cost: float
    created_at: datetime


class FeedbackRequest(BaseModel):
    helpful: bool
    notes: Optional[str] = Field(default=None, max_length=2000)
