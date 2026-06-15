"""Pydantic schemas shared across the API."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SuggestionStatus(str, Enum):
    pending = "pending"
    approved = "approved"
    rejected = "rejected"
    saved = "saved"


class Confidence(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class Suggestion(BaseModel):
    id: str
    chunk_id: str
    file: str
    breadcrumb: str
    original_text: str
    suggested_text: str
    reason: str
    confidence: Confidence
    similarity: float = 0.0
    status: SuggestionStatus = SuggestionStatus.pending
    start_line: int
    end_line: int


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)


class QueryResponse(BaseModel):
    session_id: str
    query: str
    suggestions: list[Suggestion]
    considered_chunks: int
    elapsed_ms: int = 0


class EditSuggestion(BaseModel):
    suggested_text: str


class SessionState(BaseModel):
    session_id: str
    query: str
    suggestions: list[Suggestion]
