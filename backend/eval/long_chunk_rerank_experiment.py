"""Reranking-opportunity experiment — long / hard-split chunks only.

Question: for queries whose answer lives in a LONG section chunk (hard-split, or
exactly MAX_CHARS), does the current pipeline rank that chunk inside TOP_K=12, or
does it sit in a wider pool (13-50) where a cross-encoder reranker could rescue it?

  already_top12      → expected long chunk already in top 12  (no rerank needed)
  rerank_opportunity → expected long chunk in 13-50           (rerank could help)
  retrieval_miss     → not even in top 50                     (rerank can't help)

This ONLY measures whether reranking is justified. It does not rerank, and does not
change application behaviour.

Run:
    cd backend && python -m eval.long_chunk_rerank_experiment
"""

from __future__ import annotations

import json
from pathlib import Path

from app import config
from app.chunking import MAX_CHARS, load_and_chunk
from app.retrieval import index, reciprocal_rank_fusion

WIDE_K = 50
TOP_K = config.TOP_K  # 12
HERE = Path(__file__).parent
QUERIES = json.loads((HERE / "queries.json").read_text())


def is_long_chunk_id(chunk_id: str, by_id: dict) -> bool:
    """Long if the id has a .N suffix (hard-split piece) OR text is exactly MAX_CHARS."""
    local = chunk_id.split("::", 1)[-1]
    if "." in local:  # e.g. tools.md::769.0
        return True
    c = by_id.get(chunk_id)
    return bool(c and len(c.text) == MAX_CHARS)


def setup() -> bool:
    """Build the index. Returns True if dense (embeddings) is available."""
    has_key = bool(config.OPENAI_API_KEY)
    if has_key:
        index.build()
    else:
        index.chunks = load_and_chunk(config.DOCS_ROOT)
        index.by_id = {c.chunk_id: c for c in index.chunks}
        index.bm25.fit(index.chunks)
    return has_key


def wide_pool(query: str, has_key: bool) -> list[str]:
    """Up to WIDE_K candidates. Dense+BM25+RRF if key, else BM25-only."""
    bm = [cid for cid, _ in index.bm25.search(query, WIDE_K)]
    if not has_key:
        return bm[:WIDE_K]
    dense = [c.chunk_id for c, _ in index.search(query, WIDE_K)]
    fused = [cid for cid, _ in reciprocal_rank_fusion([dense, bm])]
    return fused[:WIDE_K]


def main() -> None:
    has_key = setup()
    by_id = index.by_id
    long_ids = {c.chunk_id for c in index.chunks if is_long_chunk_id(c.chunk_id, by_id)}

    # select queries: hard-split scenario OR an expected long chunk
    selected = []
    for q in QUERIES:
        exp = q.get("expected_sections", [])
        scen = q.get("scenario", "")
        if scen in ("hard-split", "hard-split-paraphrase") or any(s in long_ids for s in exp):
            selected.append(q)

    mode = "dense+BM25+RRF" if has_key else "BM25-ONLY (no OPENAI_API_KEY — dense unavailable)"
    print(f"=== Long-chunk rerank-opportunity experiment ===")
    print(f"retrieval mode: {mode}")
    print(f"long/hard-split chunks in corpus: {len(long_ids)}")
    print(f"selected eval queries:            {len(selected)}  (wide pool K={WIDE_K}, TOP_K={TOP_K})\n")

    recall12 = recall50 = mrr50 = 0.0
    counts = {"already_top12": 0, "rerank_opportunity": 0, "retrieval_miss": 0}
    rows = []

    for q in selected:
        exp_long = [s for s in q.get("expected_sections", []) if s in long_ids]
        if not exp_long:  # selected via scenario but no labelled long section
            exp_long = [s for s in q.get("expected_sections", []) if s in long_ids] or q.get("expected_sections", [])
        pool = wide_pool(q["query"], has_key)
        pool_top12 = pool[:TOP_K]

        # rank (1-based) of first expected long chunk in the wide pool
        first_rank = None
        for s in exp_long:
            if s in pool:
                r = pool.index(s) + 1
                first_rank = r if first_rank is None else min(first_rank, r)

        in12 = any(s in pool_top12 for s in exp_long)
        in50 = first_rank is not None
        recall12 += 1.0 if in12 else 0.0
        recall50 += 1.0 if in50 else 0.0
        mrr50 += (1.0 / first_rank) if first_rank else 0.0

        if first_rank is None:
            cls = "retrieval_miss"
        elif first_rank <= TOP_K:
            cls = "already_top12"
        else:
            cls = "rerank_opportunity"
        counts[cls] += 1

        rows.append((q, exp_long, first_rank, cls, pool_top12))

    n = len(selected) or 1
    print("SUMMARY")
    print(f"  Section Recall@12 (long): {recall12 / n:.3f}")
    print(f"  Section Recall@50 (long): {recall50 / n:.3f}")
    print(f"  MRR@50 (long):            {mrr50 / n:.3f}")
    print(f"  already_top12:      {counts['already_top12']}")
    print(f"  rerank_opportunity: {counts['rerank_opportunity']}  <-- 13-50: a reranker could rescue these")
    print(f"  retrieval_miss:     {counts['retrieval_miss']}  <-- not in top 50: reranking cannot help\n")

    print("PER-QUERY DETAIL")
    for q, exp_long, rank, cls, top12 in rows:
        print(f"\n• [{q.get('scenario')}] {cls}  (first expected long rank: {rank})")
        print(f"    query: {q['query'][:78]}")
        print(f"    expected long: {exp_long}")
        print(f"    top12: {top12}")

    print("\nVERDICT")
    if counts["rerank_opportunity"] == 0 and counts["retrieval_miss"] == 0:
        print("  Long chunks are NOT a ranking problem — all expected long chunks already")
        print("  land in top 12. Cross-encoder reranking is NOT justified for long chunks.")
    elif counts["rerank_opportunity"] > 0:
        print(f"  {counts['rerank_opportunity']} query(ies) have the expected long chunk in 13-50 but")
        print("  outside top 12 — a cross-encoder reranker COULD rescue these. Reranking is")
        print("  justified for long chunks; quantify the gain before building it.")
    else:
        print("  Expected long chunks are MISSED entirely (not in top 50). That's a retrieval")
        print("  recall problem, not a ranking one — reranking won't help; fix retrieval first.")


if __name__ == "__main__":
    main()
