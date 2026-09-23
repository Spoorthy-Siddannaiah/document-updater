"""LangGraph orchestration for agentic documentation edit suggestions."""

from __future__ import annotations

import difflib
import hashlib
import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, NotRequired, TypedDict

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from . import config
from .chunking import Chunk
from .models import Confidence, Suggestion
from .observability import TraceContext, langfuse
from .persistence import json_safe, repository
from .retrieval import index, reciprocal_rank_fusion

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You maintain the OpenAI Agents SDK documentation. The user tells you what changed \
about the product or what they want to update. You are given the candidate \
documentation sections most likely affected.

Your job: decide which of these sections need editing to stay accurate and \
consistent with the user's change, and rewrite each one.

Rules:
- Only return a section if it genuinely needs a change. Do not invent edits.
- Rewrite the FULL section text (including its markdown heading line) so it can \
replace the original verbatim. Preserve markdown structure, code fences, links, \
and tables.
- Keep changes minimal and on-topic: update only what the user's change implies. \
Do not rephrase unrelated prose.
- Every edit must use a chunk_id from the candidate sections and must include \
file_path and section copied from that candidate section.
- The LLM only proposes text. It must never claim files were modified.
- Human review is mandatory before accepting any change.
- reason: one concise sentence on why this section changed.
- confidence: "high" if the section clearly must change, "medium" if probable, \
"low" if it is a judgement call.

Return JSON only, matching:
{"edits": [{"chunk_id": "<id>", "file_path": "<path>", "section": "<breadcrumb>", \
"suggested_text": "<full new section>", "reason": "<why>", "confidence": "high|medium|low"}]}
If nothing needs to change, return {"edits": []}.
"""


PROMPT_VERSION_HASH = hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()


class RetrievedChunk(BaseModel):
    chunk_id: str
    file_path: str
    section: str
    text: str
    score: float


class RawEditSuggestion(BaseModel):
    chunk_id: str
    file_path: str | None = None
    section: str | None = None
    suggested_text: str
    reason: str = ""
    confidence: str = "medium"

    @field_validator("suggested_text")
    @classmethod
    def suggested_text_is_not_empty(cls, value: str) -> str:
        value = value.strip("\n")
        if not value.strip():
            raise ValueError("suggested_text must not be empty")
        return value


class ValidatedEditSuggestion(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    chunk_id: str
    file_path: str
    section: str
    original_text: str
    suggested_text: str
    reason: str
    confidence: Confidence
    score: float
    start_line: int
    end_line: int
    status: Literal["pending_review"] = "pending_review"


class DiffLine(BaseModel):
    op: Literal["equal", "delete", "insert"]
    original: str = ""
    suggested: str = ""


class SuggestionGenerationState(TypedDict):
    run_id: str
    requested_change: str
    document_id: NotRequired[str | None]
    user_id: NotRequired[str | None]
    tenant_id: NotRequired[str | None]
    request_metadata: NotRequired[dict[str, Any]]
    model: str
    prompt_hash: str
    trace_id: NotRequired[str | None]
    trace_context: NotRequired[TraceContext]
    retrieved_chunks: list[RetrievedChunk]
    raw_edits: list[dict[str, Any]]
    suggestions: list[ValidatedEditSuggestion]
    diffs: dict[str, list[DiffLine]]
    validation_errors: list[str]
    token_usage: dict[str, Any]
    node_names: list[str]
    latency_ms: int
    status: str
    error: NotRequired[str | None]
    persisted: bool
    started_perf: NotRequired[float]


def _initial_state(
    requested_change: str,
    *,
    run_id: str | None = None,
    document_id: str | None = None,
    user_id: str | None = None,
    tenant_id: str | None = None,
    request_metadata: dict[str, Any] | None = None,
) -> SuggestionGenerationState:
    return {
        "run_id": run_id or uuid.uuid4().hex,
        "requested_change": requested_change,
        "document_id": document_id,
        "user_id": user_id,
        "tenant_id": tenant_id,
        "request_metadata": request_metadata or {},
        "model": config.CHAT_MODEL,
        "prompt_hash": PROMPT_VERSION_HASH,
        "trace_id": None,
        "retrieved_chunks": [],
        "raw_edits": [],
        "suggestions": [],
        "diffs": {},
        "validation_errors": [],
        "token_usage": {},
        "node_names": [],
        "latency_ms": 0,
        "status": "started",
        "error": None,
        "persisted": False,
        "started_perf": time.perf_counter(),
    }


def _client() -> OpenAI:
    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=config.OPENAI_API_KEY)


def _candidate_chunks(query: str) -> list[tuple[Chunk, float]]:
    dense = index.search(query, k=config.DENSE_DEPTH)
    dense_ids = [c.chunk_id for c, _ in dense]
    bm25 = index.bm25.search(query, k=config.BM25_DEPTH)
    bm25_ids = [cid for cid, _ in bm25]
    fused = reciprocal_rank_fusion([dense_ids, bm25_ids])[: config.TOP_K]

    scored: list[tuple[Chunk, float]] = []
    seen: set[str] = set()
    for cid, score in fused:
        chunk = index.by_id.get(cid)
        if chunk is not None:
            scored.append((chunk, score))
            seen.add(cid)

    for chunk in index.term_matches(query):
        if chunk.chunk_id not in seen:
            scored.append((chunk, 0.0))
            seen.add(chunk.chunk_id)
    return scored


def _render_candidates(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for chunk in chunks:
        blocks.append(
            f"<section chunk_id=\"{chunk.chunk_id}\" file_path=\"{chunk.file_path}\" "
            f"section=\"{chunk.section}\">\n{chunk.text}\n</section>"
        )
    return "\n\n".join(blocks)


def _usage_dict(resp: Any) -> dict[str, int]:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {}
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "total_tokens": getattr(usage, "total_tokens", 0) or 0,
    }


def _edits_for_batch(query: str, batch: list[RetrievedChunk]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    user_prompt = f"User change request:\n{query}\n\nCandidate sections:\n{_render_candidates(batch)}"
    resp = _client().chat.completions.create(
        model=config.CHAT_MODEL,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        return json.loads(raw).get("edits", []), _usage_dict(resp)
    except json.JSONDecodeError:
        logger.error("Model returned non-JSON: %s", raw[:500])
        return [], _usage_dict(resp)


def _merge_usage(current: dict[str, Any], extra: dict[str, int]) -> dict[str, Any]:
    merged = dict(current)
    for key, value in extra.items():
        merged[key] = int(merged.get(key, 0)) + int(value)
    return merged


def load_request_context(state: SuggestionGenerationState) -> SuggestionGenerationState:
    ctx = langfuse.start_trace(
        run_id=state["run_id"],
        user_id=state.get("user_id"),
        document_id=state.get("document_id"),
        prompt_version_hash=state["prompt_hash"],
        model=state["model"],
    )
    state["trace_context"] = ctx
    state["trace_id"] = ctx.trace_id
    with langfuse.span(ctx, "load_request_context", run_id=state["run_id"]) as span:
        if not state["requested_change"].strip():
            state["status"] = "failed"
            state["error"] = "requested_change is required"
        else:
            state["status"] = "context_loaded"
        span["document_id"] = state.get("document_id")
    return state


def retrieve_relevant_chunks(state: SuggestionGenerationState) -> SuggestionGenerationState:
    ctx = state["trace_context"]
    with langfuse.span(ctx, "retrieve_relevant_chunks") as span:
        try:
            scored = _candidate_chunks(state["requested_change"])
            state["retrieved_chunks"] = [
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    file_path=chunk.file,
                    section=chunk.breadcrumb,
                    text=chunk.text,
                    score=score,
                )
                for chunk, score in scored
            ]
            state["status"] = "retrieved" if scored else "retrieval_failed"
            span["retrieved_chunk_ids"] = [c.chunk_id for c in state["retrieved_chunks"]]
        except Exception as exc:
            state["status"] = "retrieval_failed"
            state["error"] = str(exc)
            state["validation_errors"].append(f"retrieve_relevant_chunks: {type(exc).__name__}: {exc}")
    return state


def generate_edit_suggestions(state: SuggestionGenerationState) -> SuggestionGenerationState:
    ctx = state["trace_context"]
    with langfuse.span(ctx, "generate_edit_suggestions", model=state["model"]) as span:
        if not state["retrieved_chunks"]:
            state["raw_edits"] = []
            state["status"] = "no_retrieved_chunks"
            return state
        batches = [
            state["retrieved_chunks"][i : i + config.SUGGEST_BATCH]
            for i in range(0, len(state["retrieved_chunks"]), config.SUGGEST_BATCH)
        ]
        try:
            raw_edits: list[dict[str, Any]] = []
            usage: dict[str, Any] = {}
            if len(batches) == 1:
                raw_edits, usage = _edits_for_batch(state["requested_change"], batches[0])
            else:
                with ThreadPoolExecutor(max_workers=min(len(batches), config.MAX_PARALLEL_BATCHES)) as pool:
                    for batch_edits, batch_usage in pool.map(
                        lambda batch: _edits_for_batch(state["requested_change"], batch), batches
                    ):
                        raw_edits.extend(batch_edits)
                        usage = _merge_usage(usage, batch_usage)
            state["raw_edits"] = raw_edits
            state["token_usage"] = usage
            state["status"] = "generated"
            span["tool_calls"] = [{"name": "openai.chat.completions.create", "count": len(batches)}]
            span["token_usage"] = usage
        except Exception as exc:
            state["status"] = "generation_failed"
            state["error"] = str(exc)
            state["validation_errors"].append(f"generate_edit_suggestions: {type(exc).__name__}: {exc}")
    return state


def validate_suggestions(state: SuggestionGenerationState) -> SuggestionGenerationState:
    ctx = state["trace_context"]
    with langfuse.span(ctx, "validate_suggestions") as span:
        chunks_by_id = {chunk.chunk_id: chunk for chunk in state["retrieved_chunks"]}
        validated: list[ValidatedEditSuggestion] = []
        errors: list[str] = []
        for idx, item in enumerate(state["raw_edits"]):
            try:
                raw = RawEditSuggestion.model_validate(item)
                chunk = chunks_by_id.get(raw.chunk_id)
                if chunk is None:
                    raise ValueError(f"unknown chunk_id {raw.chunk_id!r}")
                if raw.suggested_text.strip() == chunk.text.strip():
                    continue
                try:
                    confidence = Confidence(raw.confidence)
                except ValueError:
                    confidence = Confidence.medium
                validated.append(
                    ValidatedEditSuggestion(
                        chunk_id=chunk.chunk_id,
                        file_path=chunk.file_path,
                        section=chunk.section,
                        original_text=chunk.text,
                        suggested_text=raw.suggested_text.strip("\n"),
                        reason=raw.reason.strip(),
                        confidence=confidence,
                        score=chunk.score,
                        start_line=index.by_id[chunk.chunk_id].start_line,
                        end_line=index.by_id[chunk.chunk_id].end_line,
                    )
                )
            except (ValidationError, ValueError) as exc:
                errors.append(f"edit[{idx}]: {exc}")
        state["suggestions"] = validated
        state["validation_errors"].extend(errors)
        if errors and not validated:
            state["status"] = "validation_failed"
        elif errors:
            state["status"] = "validated_with_repairs"
        else:
            state["status"] = "validated"
        span["validation_errors"] = errors
        span["validated_count"] = len(validated)
    return state


def _side_by_side(original: str, suggested: str) -> list[DiffLine]:
    diff: list[DiffLine] = []
    matcher = difflib.SequenceMatcher(a=original.splitlines(), b=suggested.splitlines())
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        old = original.splitlines()[i1:i2]
        new = suggested.splitlines()[j1:j2]
        if tag == "equal":
            diff.extend(DiffLine(op="equal", original=line, suggested=line) for line in old)
        elif tag == "delete":
            diff.extend(DiffLine(op="delete", original=line) for line in old)
        elif tag == "insert":
            diff.extend(DiffLine(op="insert", suggested=line) for line in new)
        elif tag == "replace":
            diff.extend(DiffLine(op="delete", original=line) for line in old)
            diff.extend(DiffLine(op="insert", suggested=line) for line in new)
    return diff


def generate_side_by_side_diff(state: SuggestionGenerationState) -> SuggestionGenerationState:
    ctx = state["trace_context"]
    with langfuse.span(ctx, "generate_side_by_side_diff") as span:
        state["diffs"] = {
            suggestion.id: _side_by_side(suggestion.original_text, suggestion.suggested_text)
            for suggestion in state["suggestions"]
        }
        if state["status"] in {"validated", "validated_with_repairs"}:
            state["status"] = "diffed"
        span["diff_count"] = len(state["diffs"])
    return state


def persist_suggestions(state: SuggestionGenerationState) -> SuggestionGenerationState:
    ctx = state["trace_context"]
    with langfuse.span(ctx, "persist_suggestions") as span:
        state["latency_ms"] = int((time.perf_counter() - state.get("started_perf", time.perf_counter())) * 1000)
        final_status = "completed" if state["status"] == "diffed" else state["status"]
        run_record = {
            "run_id": state["run_id"],
            "document_id": state.get("document_id"),
            "user_id": state.get("user_id"),
            "tenant_id": state.get("tenant_id"),
            "requested_change": state["requested_change"],
            "prompt_hash": state["prompt_hash"],
            "model": state["model"],
            "trace_id": state.get("trace_id"),
            "retrieved_chunk_ids": [chunk.chunk_id for chunk in state["retrieved_chunks"]],
            "node_names": state.get("trace_context", TraceContext()).node_names,
            "latency_ms": state["latency_ms"],
            "status": final_status,
            "token_usage": state["token_usage"],
            "validation_errors": state["validation_errors"],
        }
        suggestion_records = []
        for suggestion in state["suggestions"]:
            suggestion_records.append(
                {
                    "id": suggestion.id,
                    "run_id": state["run_id"],
                    "chunk_id": suggestion.chunk_id,
                    "file_path": suggestion.file_path,
                    "section": suggestion.section,
                    "original_text": suggestion.original_text,
                    "suggested_text": suggestion.suggested_text,
                    "reason": suggestion.reason,
                    "confidence": suggestion.confidence.value
                    if isinstance(suggestion.confidence, Confidence)
                    else str(suggestion.confidence),
                    "score": suggestion.score,
                    "diff": [line.model_dump() for line in state["diffs"].get(suggestion.id, [])],
                    "status": suggestion.status,
                }
            )
        try:
            state["persisted"] = repository.persist_suggestion_run(
                run=json_safe(run_record),
                suggestions=json_safe(suggestion_records),
            )
        except Exception as exc:
            state["persisted"] = False
            state["validation_errors"].append(f"persist_suggestions: {type(exc).__name__}: {exc}")
            if state["status"] == "diffed":
                state["status"] = "persist_failed"
        span["persisted"] = state["persisted"]
    return state


def return_result(state: SuggestionGenerationState) -> SuggestionGenerationState:
    ctx = state["trace_context"]
    with langfuse.span(ctx, "return_result") as span:
        state["latency_ms"] = int((time.perf_counter() - state.get("started_perf", time.perf_counter())) * 1000)
        if state["status"] == "diffed":
            state["status"] = "completed"
        state["node_names"] = ctx.node_names
        langfuse.update_trace(
            ctx,
            run_id=state["run_id"],
            user_id=state.get("user_id"),
            document_id=state.get("document_id"),
            prompt_version_hash=state["prompt_hash"],
            model=state["model"],
            retrieved_chunk_ids=[chunk.chunk_id for chunk in state["retrieved_chunks"]],
            node_names=state["node_names"],
            latency_ms=state["latency_ms"],
            token_usage=state["token_usage"],
            validation_errors=state["validation_errors"],
            final_status=state["status"],
        )
        span["final_status"] = state["status"]
    langfuse.flush()
    return state


def _can_generate(state: SuggestionGenerationState) -> str:
    return "generate" if state["status"] == "retrieved" else "fallback"


def _can_diff(state: SuggestionGenerationState) -> str:
    return "diff" if state["suggestions"] else "fallback"


class AgenticSuggestionWorkflow:
    def __init__(self) -> None:
        self._graph = self._build_graph()

    def _build_graph(self) -> Any | None:
        try:
            from langgraph.graph import END, StateGraph
        except Exception:
            return None

        graph = StateGraph(SuggestionGenerationState)
        graph.add_node("load_request_context", load_request_context)
        graph.add_node("retrieve_relevant_chunks", retrieve_relevant_chunks)
        graph.add_node("generate_edit_suggestions", generate_edit_suggestions)
        graph.add_node("validate_suggestions", validate_suggestions)
        graph.add_node("generate_side_by_side_diff", generate_side_by_side_diff)
        graph.add_node("persist_suggestions", persist_suggestions)
        graph.add_node("return_result", return_result)

        graph.set_entry_point("load_request_context")
        graph.add_edge("load_request_context", "retrieve_relevant_chunks")
        graph.add_conditional_edges(
            "retrieve_relevant_chunks",
            _can_generate,
            {"generate": "generate_edit_suggestions", "fallback": "persist_suggestions"},
        )
        graph.add_edge("generate_edit_suggestions", "validate_suggestions")
        graph.add_conditional_edges(
            "validate_suggestions",
            _can_diff,
            {"diff": "generate_side_by_side_diff", "fallback": "persist_suggestions"},
        )
        graph.add_edge("generate_side_by_side_diff", "persist_suggestions")
        graph.add_edge("persist_suggestions", "return_result")
        graph.add_edge("return_result", END)
        return graph.compile()

    def run(
        self,
        requested_change: str,
        *,
        run_id: str | None = None,
        document_id: str | None = None,
        user_id: str | None = None,
        tenant_id: str | None = None,
        request_metadata: dict[str, Any] | None = None,
    ) -> SuggestionGenerationState:
        started = time.perf_counter()
        state = _initial_state(
            requested_change,
            run_id=run_id,
            document_id=document_id,
            user_id=user_id,
            tenant_id=tenant_id,
            request_metadata=request_metadata,
        )
        if self._graph is not None:
            result = self._graph.invoke(state)
        else:
            result = self._run_without_langgraph(state)
        result["latency_ms"] = int((time.perf_counter() - started) * 1000)
        return result

    @staticmethod
    def _run_without_langgraph(state: SuggestionGenerationState) -> SuggestionGenerationState:
        state = load_request_context(state)
        state = retrieve_relevant_chunks(state)
        if state["status"] == "retrieved":
            state = generate_edit_suggestions(state)
            state = validate_suggestions(state)
            if state["suggestions"]:
                state = generate_side_by_side_diff(state)
        state = persist_suggestions(state)
        state = return_result(state)
        return state

    @staticmethod
    def to_api_suggestions(state: SuggestionGenerationState) -> list[Suggestion]:
        return [
            Suggestion(
                id=suggestion.id,
                chunk_id=suggestion.chunk_id,
                file=suggestion.file_path,
                breadcrumb=suggestion.section,
                original_text=suggestion.original_text,
                suggested_text=suggestion.suggested_text,
                reason=suggestion.reason,
                confidence=suggestion.confidence,
                similarity=suggestion.score,
                start_line=suggestion.start_line,
                end_line=suggestion.end_line,
            )
            for suggestion in state["suggestions"]
        ]


workflow = AgenticSuggestionWorkflow()
