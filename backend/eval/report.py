"""One eval report — all primary + secondary metrics in a single table.

Measures the ACTUAL production pipeline (RRF + term-match, the candidate set the
LLM really sees), not isolated retriever variants.

RETRIEVAL  (over queries.json, needs embeddings)
  File Hit@5        coarse sanity — right file in top-5?
  Section Hit@5     quick proxy — at least one right section in top-5?
  Section Recall    PRIMARY — of all expected sections, how many reached the LLM?
  MRR@5             ranking quality — how early is the first correct section?
  Avg candidates    cost / context pressure

SUGGESTION  (over suggestion_queries.json, needs LLM calls — slower)
  Suggestion Recall PRIMARY — of expected target sections, how many were edited?
  Assertion pass    must_mention / must_not_keep / must_not_mention all satisfied?
  No-op Accuracy    on junk/vague queries, did it return zero edits?

Run:
    cd backend && python -m eval.report              # retrieval only (fast)
    cd backend && python -m eval.report --suggestions # + LLM suggestion metrics
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from app import config
from app.chunking import load_and_chunk
from app.retrieval import index, reciprocal_rank_fusion

HIT_K = 5
HERE = Path(__file__).parent
RET_Q = json.loads((HERE / "queries.json").read_text())
SUG_Q = json.loads((HERE / "suggestion_queries.json").read_text())


def setup() -> None:
    if config.OPENAI_API_KEY:
        index.build()
    else:
        index.chunks = load_and_chunk(config.DOCS_ROOT)
        index.by_id = {c.chunk_id: c for c in index.chunks}
        index.bm25.fit(index.chunks)


# ---- retrieval (the real pipeline candidate set) ----------------------------
def ranked_topk(query: str) -> list[str]:
    d = [c.chunk_id for c, _ in index.search(query, config.DENSE_DEPTH)]
    b = [cid for cid, _ in index.bm25.search(query, config.BM25_DEPTH)]
    return [cid for cid, _ in reciprocal_rank_fusion([d, b])][: config.TOP_K]


def full_candidates(query: str) -> list[str]:
    ids = ranked_topk(query)
    seen = set(ids)
    for c in index.term_matches(query):
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            ids.append(c.chunk_id)
    return ids


def retrieval_report() -> None:
    queries = [q for q in RET_Q if q["expected_files"]]
    n = len(queries)
    file_hit = sec_hit = sec_recall = mrr = cand = 0.0
    n_sec = 0

    for q in queries:
        ranked = ranked_topk(q["query"])[:HIT_K]
        cands = full_candidates(q["query"])
        cand += len(cands)

        exp_files = set(q["expected_files"])
        got_files = {index.by_id[c].file for c in ranked if c in index.by_id}
        file_hit += 1.0 if exp_files & got_files else 0.0

        exp_secs = set(q.get("expected_sections", []))
        if exp_secs:
            n_sec += 1
            top = set(ranked)
            sec_hit += 1.0 if exp_secs & top else 0.0
            # recall over the FULL candidate set (what the LLM actually sees)
            sec_recall += len(exp_secs & set(cands)) / len(exp_secs)
            # MRR over ranked top-K
            rr = 0.0
            for rank, cid in enumerate(ranked):
                if cid in exp_secs:
                    rr = 1.0 / (rank + 1)
                    break
            mrr += rr

    print("RETRIEVAL  (production pipeline: RRF + term-match)")
    print(f"  queries:            {n}  ({n_sec} section-labelled)")
    print(f"  File Hit@{HIT_K}:          {file_hit / n:.3f}")
    print(f"  Section Hit@{HIT_K}:       {sec_hit / n_sec:.3f}")
    print(f"  Section Recall:     {sec_recall / n_sec:.3f}   <-- PRIMARY (found all places)")
    print(f"  MRR@{HIT_K}:              {mrr / n_sec:.3f}")
    print(f"  Avg candidates:     {cand / n:.1f}")


# ---- suggestion (LLM) -------------------------------------------------------
def _score_query(q) -> dict:
    from app.suggest import generate_suggestions

    sugs, _ = generate_suggestions(q["query"])
    edited_files = {s.file for s in sugs}
    edited_secs = {s.chunk_id for s in sugs}
    new_text = " ".join(s.suggested_text for s in sugs)

    if not q.get("target_sections"):  # no-op / vague
        return {"noop": 1.0 if len(sugs) <= q.get("expected_max_edits", 0) else 0.0}

    tgt_secs = set(q["target_sections"])
    tgt_files = {s.split("::")[0] for s in tgt_secs}
    ok = (
        all(m in new_text for m in q.get("must_mention", []))
        and all(k not in new_text for k in q.get("must_not_keep", []))
        and all(k not in new_text for k in q.get("must_not_mention", []))
        and not (edited_files & {f for f in q.get("must_not_edit", []) if f != "*"})
    )
    return {
        "recall_file": len(tgt_files & edited_files) / len(tgt_files),
        "recall_sec": len(tgt_secs & edited_secs) / len(tgt_secs) if tgt_secs else 0.0,
        "assert": 1.0 if ok else 0.0,
    }


def suggestion_report() -> None:
    from collections import defaultdict

    by_type: dict[str, list[dict]] = defaultdict(list)
    for q in SUG_Q:
        by_type[q.get("type", "?")].append(_score_query(q))

    print("\nSUGGESTION  (live LLM, scored per query-type)")
    print(f"  {'type':<20}{'n':>3}{'Recall(file)':>14}{'Recall(sec)':>13}{'Assert':>9}{'No-op':>8}")
    print("  " + "-" * 67)

    def avg(rows, key):
        vals = [r[key] for r in rows if key in r]
        return sum(vals) / len(vals) if vals else None

    fmt = lambda x: f"{x:.2f}" if x is not None else "  -"
    order = ["exact-symbol", "natural-paraphrase", "number-value",
             "cross-cutting", "code-block", "no-op-vague"]
    for t in order:
        rows = by_type.get(t, [])
        if not rows:
            continue
        rf, rs, ac = avg(rows, "recall_file"), avg(rows, "recall_sec"), avg(rows, "assert")
        noop = avg(rows, "noop")
        noop_s = f"{sum(r.get('noop',0) for r in rows):.0f}/{len(rows)}" if noop is not None else "  -"
        print(f"  {t:<20}{len(rows):>3}{fmt(rf):>14}{fmt(rs):>13}{fmt(ac):>9}{noop_s:>8}")


def main() -> None:
    setup()
    print(f"=== EVAL REPORT | {len(index.chunks)} chunks | K={config.TOP_K} ===\n")
    retrieval_report()
    if "--suggestions" in sys.argv:
        if not config.OPENAI_API_KEY:
            print("\n(suggestion metrics need OPENAI_API_KEY)")
        else:
            suggestion_report()
    else:
        print("\n(run with --suggestions for LLM suggestion metrics)")


if __name__ == "__main__":
    main()
