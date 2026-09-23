"""Small Langfuse wrapper used by the agentic workflow.

The rest of the app calls this module instead of the Langfuse SDK directly so
local development still works when Langfuse credentials are not configured.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from . import config


@dataclass
class TraceContext:
    trace: Any | None = None
    trace_id: str | None = None
    node_names: list[str] = field(default_factory=list)


class LangfuseClient:
    def __init__(self) -> None:
        self._client: Any | None = None
        if not config.LANGFUSE_ENABLED:
            return
        try:
            from langfuse import Langfuse

            self._client = Langfuse(
                public_key=config.LANGFUSE_PUBLIC_KEY or None,
                secret_key=config.LANGFUSE_SECRET_KEY or None,
                host=config.LANGFUSE_HOST,
            )
        except Exception:
            self._client = None

    def start_trace(
        self,
        *,
        run_id: str,
        user_id: str | None,
        document_id: str | None,
        prompt_version_hash: str,
        model: str,
    ) -> TraceContext:
        metadata = {
            "run_id": run_id,
            "document_id": document_id,
            "prompt_version_hash": prompt_version_hash,
            "model": model,
        }
        if self._client is None:
            return TraceContext(trace_id=None)
        if hasattr(self._client, "trace"):
            trace = self._client.trace(
                id=run_id,
                name="suggestion_generation",
                user_id=user_id,
                metadata=metadata,
            )
            return TraceContext(trace=trace, trace_id=getattr(trace, "id", run_id))
        trace_id = None
        if hasattr(self._client, "create_trace_id"):
            try:
                trace_id = self._client.create_trace_id(seed=run_id)
            except Exception:
                trace_id = run_id
        return TraceContext(trace=self._client, trace_id=trace_id or run_id)

    @contextmanager
    def span(self, ctx: TraceContext, name: str, **metadata: Any) -> Iterator[dict[str, Any]]:
        ctx.node_names.append(name)
        started = time.perf_counter()
        span = None
        otel_cm = None
        if ctx.trace is not None and hasattr(ctx.trace, "span"):
            try:
                span = ctx.trace.span(name=name, metadata=metadata)
            except Exception:
                span = None
        elif ctx.trace is not None and hasattr(ctx.trace, "_otel_tracer"):
            attrs = {f"app.{key}": value for key, value in metadata.items() if isinstance(value, str | int | float | bool)}
            try:
                otel_cm = ctx.trace._otel_tracer.start_as_current_span(name, attributes=attrs)
                span = otel_cm.__enter__()
                ctx.trace.update_current_span(
                    name=name,
                    metadata={**metadata, "trace_id": ctx.trace_id},
                )
            except Exception:
                span = None
                otel_cm = None
        payload: dict[str, Any] = {}
        try:
            yield payload
            status = "ok"
        except Exception as exc:
            status = "error"
            payload["error_type"] = type(exc).__name__
            payload["error"] = str(exc)
            raise
        finally:
            latency_ms = int((time.perf_counter() - started) * 1000)
            output = {"status": status, "latency_ms": latency_ms, **payload}
            if otel_cm is not None:
                try:
                    ctx.trace.update_current_span(output=output)
                    otel_cm.__exit__(None, None, None)
                except Exception:
                    pass
            elif span is not None:
                try:
                    span.end(output=output)
                except Exception:
                    pass

    def update_trace(self, ctx: TraceContext, **metadata: Any) -> None:
        if ctx.trace is None:
            return
        try:
            ctx.trace.update(metadata=metadata)
        except Exception:
            pass

    def flush(self) -> None:
        if self._client is None:
            return
        try:
            self._client.flush()
        except Exception:
            pass


langfuse = LangfuseClient()
