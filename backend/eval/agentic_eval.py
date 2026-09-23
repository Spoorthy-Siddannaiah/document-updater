"""Agentic suggestion eval harness.

Runs a fixed documentation-change dataset through the LangGraph suggestion
workflow and records comparable metrics for prompt/model experiments.

Run:
    cd backend && python -m eval.agentic_eval
"""

from __future__ import annotations

import difflib
import json
import time
import uuid
from pathlib import Path
from typing import Any

from app import config
from app.agentic_workflow import PROMPT_VERSION_HASH, workflow
from app.chunking import load_and_chunk
from app.observability import langfuse
from app.persistence import json_safe, repository
from app.retrieval import index

DATASET = json.loads((Path(__file__).parent / "suggestion_queries.json").read_text())


def _source_by_file() -> dict[str, str]:
    return {
        chunk.file: (config.DOCS_ROOT / chunk.file).read_text(encoding="utf-8")
        for chunk in load_and_chunk(config.DOCS_ROOT)
    }


def _diff_similarity(suggestions: list[Any], item: dict[str, Any]) -> float:
    expected_terms = " ".join(item.get("must_mention", []) + item.get("must_not_keep", []))
    actual = "\n".join(s.suggested_text for s in suggestions)
    if not expected_terms.strip() and not actual.strip():
        return 1.0
    if not expected_terms.strip() or not actual.strip():
        return 0.0
    return difflib.SequenceMatcher(a=expected_terms.lower(), b=actual.lower()).ratio()


def score_item(item: dict[str, Any], suggestions: list[Any], latency_ms: int) -> dict[str, Any]:
    target_sections = set(item.get("target_sections", []))
    target_files = {target.split("::")[0] for target in target_sections}
    suggested_chunks = {s.chunk_id for s in suggestions}
    suggested_files = {s.file for s in suggestions}
    source = _source_by_file()

    file_hits = len(target_files & suggested_files)
    section_hits = len(target_sections & suggested_chunks)
    hallucinated = [
        s.id for s in suggestions
        if not s.original_text.strip() or s.original_text not in source.get(s.file, "")
    ]
    forbidden_files = set(item.get("must_not_edit", []))
    if "*" in forbidden_files:
        false_positive_files = suggested_files
    else:
        false_positive_files = forbidden_files & suggested_files

    precision = 1.0 if not suggestions else 1 - (len(false_positive_files) / max(len(suggested_files), 1))
    recall = 1.0 if not target_sections else section_hits / len(target_sections)

    return {
        "file_path_match": 1.0 if not target_files else file_hits / len(target_files),
        "section_chunk_match": 1.0 if not target_sections else section_hits / len(target_sections),
        "diff_similarity": _diff_similarity(suggestions, item),
        "suggestion_precision": max(0.0, precision),
        "recall_against_real_git_changes": recall,
        "hallucinated_edit_rate": 0.0 if not suggestions else len(hallucinated) / len(suggestions),
        "validation_failure_rate": 0.0,
        "latency_ms": latency_ms,
        "estimated_cost": None,
        "suggestion_count": len(suggestions),
    }


def run_eval(dataset_name: str = "suggestion_queries") -> dict[str, Any]:
    if index.matrix is None:
        index.build()

    eval_run_id = uuid.uuid4().hex
    started = time.perf_counter()
    trace = langfuse.start_trace(
        run_id=eval_run_id,
        user_id=None,
        document_id=f"eval:{dataset_name}",
        prompt_version_hash=PROMPT_VERSION_HASH,
        model=config.CHAT_MODEL,
    )
    result_records: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []

    with langfuse.span(trace, "run_eval_dataset", dataset_name=dataset_name) as span:
        for idx, item in enumerate(DATASET):
            t0 = time.perf_counter()
            state = workflow.run(
                item["query"],
                document_id=f"eval:{dataset_name}",
                request_metadata={"scenario": item.get("scenario")},
            )
            suggestions = workflow.to_api_suggestions(state)
            latency_ms = int((time.perf_counter() - t0) * 1000)
            item_metrics = score_item(item, suggestions, latency_ms)
            item_metrics["validation_failure_rate"] = 1.0 if state["validation_errors"] and not suggestions else 0.0
            if state["token_usage"]:
                item_metrics["token_usage"] = state["token_usage"]
            metrics.append(item_metrics)
            result_records.append(
                {
                    "id": f"{eval_run_id}:{idx}",
                    "eval_run_id": eval_run_id,
                    "scenario": item.get("scenario", "unknown"),
                    "query": item["query"],
                    "metrics": item_metrics,
                    "latency_ms": latency_ms,
                    "estimated_cost": item_metrics["estimated_cost"],
                }
            )

        summary = {
            key: sum(float(m.get(key, 0) or 0) for m in metrics) / max(len(metrics), 1)
            for key in (
                "file_path_match",
                "section_chunk_match",
                "diff_similarity",
                "suggestion_precision",
                "recall_against_real_git_changes",
                "hallucinated_edit_rate",
                "validation_failure_rate",
                "latency_ms",
            )
        }
        span["summary"] = summary

    total_latency_ms = int((time.perf_counter() - started) * 1000)
    eval_record = {
        "eval_run_id": eval_run_id,
        "dataset_name": dataset_name,
        "prompt_hash": PROMPT_VERSION_HASH,
        "model": config.CHAT_MODEL,
        "trace_id": trace.trace_id,
        "summary": summary,
        "status": "completed",
        "latency_ms": total_latency_ms,
    }
    persisted = repository.persist_eval_run(
        eval_run=json_safe(eval_record),
        results=json_safe(result_records),
    )
    langfuse.update_trace(
        trace,
        eval_run_id=eval_run_id,
        prompt_version_hash=PROMPT_VERSION_HASH,
        model=config.CHAT_MODEL,
        final_status="completed",
        latency_ms=total_latency_ms,
        persisted=persisted,
    )
    langfuse.flush()
    return {"eval_run": eval_record, "summary": summary, "persisted": persisted}


def main() -> None:
    result = run_eval()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
