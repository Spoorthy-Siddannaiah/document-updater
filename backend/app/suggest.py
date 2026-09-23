"""Suggestion engine: turn a natural-language change request into concrete,
section-level edit proposals.

Pipeline: retrieve the top-K candidate chunks for the query, hand them to the chat
model with their ids, and ask it to return — as strict JSON — only the chunks that
actually need editing, each with a full replacement body, a reason, and a
confidence. We re-attach the original text and line span from our own chunk store
(never trusting the model for those) so an approved edit maps back to exact lines.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from . import config
from .agentic_workflow import workflow
from .chunking import Chunk
from .models import Confidence, Suggestion
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
- If a section describes a now-removed feature, update or remove that guidance and, \
where relevant, point readers to the recommended alternative.
- reason: one concise sentence on why this section changed.
- confidence: "high" if the section clearly must change, "medium" if probable, \
"low" if it is a judgement call.

Return JSON only, matching:
{"edits": [{"chunk_id": "<id>", "suggested_text": "<full new section>", \
"reason": "<why>", "confidence": "high|medium|low"}]}
If nothing needs to change, return {"edits": []}.
"""


def _render_candidates(scored: list[tuple[Chunk, float]]) -> str:
    blocks = []
    for chunk, _score in scored:
        blocks.append(
            f"<section chunk_id=\"{chunk.chunk_id}\" file=\"{chunk.file}\" "
            f"location=\"{chunk.breadcrumb}\">\n{chunk.text}\n</section>"
        )
    return "\n\n".join(blocks)


def _client() -> OpenAI:
    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=config.OPENAI_API_KEY)


def _candidate_chunks(query: str) -> list[tuple[Chunk, float]]:
    """Hybrid retrieval: RRF-fuse dense + BM25, then force-include symbol hits.

    - Dense (embeddings) finds semantically-related prose.
    - BM25 finds exact-wording lexical matches the embedding may not surface.
    - These two ranked lists are fused with Reciprocal Rank Fusion (rank-based, so
      their incomparable raw scores merge fairly) and trimmed to TOP_K.
    - Symbol grep then force-includes any code section naming an exact identifier
      from the query (e.g. ``as_tool``) that fusion left out — a pure recall floor.

    Returned floats are the fused relevance score (RRF), used only as a UI tiebreak.
    """
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

    # Completeness floor for cross-cutting changes ("rename X everywhere"). The
    # capped RRF list structurally cannot cover a term that appears in many places
    # (measured: 'handoff' recall 0.14 capped vs 0.93 with this union). So we union
    # every chunk containing an exact query symbol — uncapped.
    for chunk in index.term_matches(query):
        if chunk.chunk_id not in seen:
            seen.add(chunk.chunk_id)
            scored.append((chunk, 0.0))
    return scored


def _edits_for_batch(
    query: str, batch: list[tuple[Chunk, float]], temperature: float = 0.2
) -> list[dict]:
    """One LLM call over a batch of candidate sections; returns raw edit dicts."""
    user_prompt = (
        f"User change request:\n{query}\n\n"
        f"Candidate sections:\n{_render_candidates(batch)}"
    )
    resp = _client().chat.completions.create(
        model=config.CHAT_MODEL,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        return json.loads(raw).get("edits", [])
    except json.JSONDecodeError:
        logger.error("Model returned non-JSON: %s", raw[:500])
        return []


def regenerate_one(query: str, chunk: Chunk) -> dict | None:
    """Re-run the LLM on a single section. Returns a fresh edit dict, or None if
    the model now judges no change is needed. Slightly higher temperature so a
    re-roll actually differs from the first attempt."""
    edits = _edits_for_batch(query, [(chunk, 0.0)], temperature=0.5)
    for e in edits:
        if e.get("chunk_id") == chunk.chunk_id and (e.get("suggested_text") or "").strip():
            return e
    return None


def generate_suggestions(query: str) -> tuple[list[Suggestion], int]:
    t_start = time.perf_counter()
    state = workflow.run(query)
    suggestions = workflow.to_api_suggestions(state)
    order = {Confidence.high: 0, Confidence.medium: 1, Confidence.low: 2}
    suggestions.sort(key=lambda s: (order[s.confidence], -s.similarity))
    considered = len(state["retrieved_chunks"])
    logger.info(
        "suggest query=%r total=%.0fms candidates=%d edits=%d status=%s trace_id=%s",
        query[:40],
        (time.perf_counter() - t_start) * 1000,
        considered,
        len(suggestions),
        state["status"],
        state.get("trace_id"),
    )
    return suggestions, considered
