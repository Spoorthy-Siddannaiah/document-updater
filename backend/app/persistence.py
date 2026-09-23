"""PostgreSQL persistence for agent runs and evaluation results.

PostgreSQL is the durable source of truth in production. The repository is
disabled when DATABASE_URL or SQLAlchemy are unavailable so the local app remains
runnable without infrastructure.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from . import config

logger = logging.getLogger(__name__)

try:
    from sqlalchemy import JSON, DateTime, Float, Integer, String, Text, create_engine
    from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column
except Exception:  # pragma: no cover - optional production dependency
    JSON = DateTime = Float = Integer = String = Text = create_engine = None
    DeclarativeBase = Mapped = Session = mapped_column = None


if DeclarativeBase is not None:

    class Base(DeclarativeBase):
        pass

    class SuggestionRunRecord(Base):
        """One durable record per agentic suggestion-generation run."""

        __tablename__ = "suggestion_runs"

        run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
        document_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
        user_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
        tenant_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
        requested_change: Mapped[str] = mapped_column(Text)
        prompt_hash: Mapped[str] = mapped_column(String(64), index=True)
        model: Mapped[str] = mapped_column(String(128))
        trace_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
        retrieved_chunk_ids: Mapped[list[str]] = mapped_column(JSON)
        node_names: Mapped[list[str]] = mapped_column(JSON)
        latency_ms: Mapped[int] = mapped_column(Integer)
        status: Mapped[str] = mapped_column(String(32), index=True)
        token_usage: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
        validation_errors: Mapped[list[str]] = mapped_column(JSON)
        created_at: Mapped[datetime] = mapped_column(
            DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
        )

    class EditSuggestionRecord(Base):
        """Validated model suggestion mapped back to trusted local chunk metadata."""

        __tablename__ = "edit_suggestions"

        id: Mapped[str] = mapped_column(String(64), primary_key=True)
        run_id: Mapped[str] = mapped_column(String(64), index=True)
        chunk_id: Mapped[str] = mapped_column(String(512), index=True)
        file_path: Mapped[str] = mapped_column(String(512), index=True)
        section: Mapped[str] = mapped_column(String(512))
        original_text: Mapped[str] = mapped_column(Text)
        suggested_text: Mapped[str] = mapped_column(Text)
        reason: Mapped[str] = mapped_column(Text)
        confidence: Mapped[str] = mapped_column(String(16))
        score: Mapped[float] = mapped_column(Float)
        diff: Mapped[list[dict[str, str]]] = mapped_column(JSON)
        status: Mapped[str] = mapped_column(String(32), default="pending_review")
        created_at: Mapped[datetime] = mapped_column(
            DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
        )

    class EvalRunRecord(Base):
        """A comparable eval execution for a dataset, prompt hash, and model."""

        __tablename__ = "eval_runs"

        eval_run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
        dataset_name: Mapped[str] = mapped_column(String(128))
        prompt_hash: Mapped[str] = mapped_column(String(64), index=True)
        model: Mapped[str] = mapped_column(String(128), index=True)
        trace_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
        summary: Mapped[dict[str, Any]] = mapped_column(JSON)
        status: Mapped[str] = mapped_column(String(32), index=True)
        latency_ms: Mapped[int] = mapped_column(Integer)
        created_at: Mapped[datetime] = mapped_column(
            DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
        )

    class EvalResultRecord(Base):
        """Per-task evaluation metrics for retrieval, diff quality, and safety."""

        __tablename__ = "eval_results"

        id: Mapped[str] = mapped_column(String(96), primary_key=True)
        eval_run_id: Mapped[str] = mapped_column(String(64), index=True)
        scenario: Mapped[str] = mapped_column(String(128), index=True)
        query: Mapped[str] = mapped_column(Text)
        metrics: Mapped[dict[str, Any]] = mapped_column(JSON)
        latency_ms: Mapped[int] = mapped_column(Integer)
        estimated_cost: Mapped[float | None] = mapped_column(Float, nullable=True)

else:
    Base = None
    SuggestionRunRecord = EditSuggestionRecord = EvalRunRecord = EvalResultRecord = None


class PostgresRepository:
    def __init__(self) -> None:
        self.enabled = bool(config.DATABASE_URL and create_engine is not None and Base is not None)
        self._engine = None
        if not self.enabled:
            return
        self._engine = create_engine(config.DATABASE_URL, pool_pre_ping=True)
        Base.metadata.create_all(self._engine)

    def persist_suggestion_run(
        self,
        *,
        run: dict[str, Any],
        suggestions: list[dict[str, Any]],
    ) -> bool:
        if not self.enabled or self._engine is None:
            logger.info("PostgreSQL persistence skipped; DATABASE_URL is not configured")
            return False
        with Session(self._engine) as session:
            session.merge(SuggestionRunRecord(**run))
            for suggestion in suggestions:
                session.merge(EditSuggestionRecord(**suggestion))
            session.commit()
        return True

    def persist_eval_run(
        self,
        *,
        eval_run: dict[str, Any],
        results: list[dict[str, Any]],
    ) -> bool:
        if not self.enabled or self._engine is None:
            logger.info("Eval persistence skipped; DATABASE_URL is not configured")
            return False
        with Session(self._engine) as session:
            session.merge(EvalRunRecord(**eval_run))
            for result in results:
                session.merge(EvalResultRecord(**result))
            session.commit()
        return True


repository = PostgresRepository()


def json_safe(value: Any) -> Any:
    """Round-trip values through JSON so PostgreSQL JSON columns get primitives."""
    return json.loads(json.dumps(value, default=str))
