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


class SuggestionGenerationRequest(BaseModel):
    requested_change: str = Field(min_length=1)
    document_id: str | None = None
    user_id: str | None = None
    tenant_id: str | None = None
    run_id: str | None = None


class RetrievedChunkResponse(BaseModel):
    chunk_id: str
    file_path: str
    section: str
    text: str
    score: float


class DiffLineResponse(BaseModel):
    op: str
    original: str = ""
    suggested: str = ""


class AgenticSuggestionResponse(BaseModel):
    run_id: str
    trace_id: str | None = None
    document_id: str | None = None
    requested_change: str
    suggestions: list[Suggestion]
    retrieved_chunks: list[RetrievedChunkResponse]
    diffs: dict[str, list[DiffLineResponse]]
    retrieved_chunk_ids: list[str]
    prompt_hash: str
    model: str
    latency_ms: int
    token_usage: dict
    validation_errors: list[str]
    node_names: list[str]
    status: str
    persisted: bool


class EditSuggestion(BaseModel):
    suggested_text: str


class SessionState(BaseModel):
    session_id: str
    query: str
    suggestions: list[Suggestion]
