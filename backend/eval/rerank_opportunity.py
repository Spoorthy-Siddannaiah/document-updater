"""Rerank-opportunity experiment — whole eval set.

Reranking only helps when an expected section is retrieved into a WIDER pool but
ranked below the TOP_K=12 the LLM sees. This measures exactly that, per expected
section:

  already_top12      rank <= 12         reranking unnecessary
  rerank_opportunity rank 13-50         a cross-encoder COULD lift it into the LLM's view
  miss               not in top 50      reranking can't help — retrieval recall gap

Headline comparison: Section Recall@12 vs Section Recall@50. The gap between them
is the ceiling reranking could recover.

Run:
    cd backend && python -m eval.rerank_opportunity
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from app import config
from app.chunking import load_and_chunk
from app.retrieval import index, reciprocal_rank_fusion

WIDE_K = 50
TOP_K = config.TOP_K
QUERIES = [q for q in json.loads((Path(__file__).parent / "queries.json").read_text())
           if q.get("expected_sections")]


def setup() -> bool:
    has_key = bool(config.OPENAI_API_KEY)
    if has_key:
        index.build()
    else:
        index.chunks = load_and_chunk(config.DOCS_ROOT)
        index.by_id = {c.chunk_id: c for c in index.chunks}
        index.bm25.fit(index.chunks)
    return has_key


def wide_pool(query: str, has_key: bool) -> list[str]:
    bm = [cid for cid, _ in index.bm25.search(query, WIDE_K)]
    if not has_key:
        return bm[:WIDE_K]
    dense = [c.chunk_id for c, _ in index.search(query, WIDE_K)]
    return [cid for cid, _ in reciprocal_rank_fusion([dense, bm])][:WIDE_K]


def main() -> None:
    has_key = setup()
    mode = "dense+BM25+RRF" if has_key else "BM25-ONLY (no key)"
    print(f"=== Rerank-opportunity (whole eval set) | mode: {mode} ===")
    print(f"queries: {len(QUERIES)} | wide pool K={WIDE_K} | TOP_K={TOP_K}\n")

    # per-section classification + per-query recall
    sec_cls = Counter()
    recall12_sum = recall50_sum = 0.0
    rescuable_queries = []  # queries where >0 targets are in 13-50

    print(f"{'scenario':<18}{'R@12':>6}{'R@50':>6}  query")
    print("-" * 74)
    for q in QUERIES:
        pool = wide_pool(q["query"], has_key)
        rank_of = {cid: i + 1 for i, cid in enumerate(pool)}
        exp = q["expected_sections"]

        in12 = sum(1 for s in exp if rank_of.get(s, 999) <= TOP_K)
        in50 = sum(1 for s in exp if s in rank_of)
        r12, r50 = in12 / len(exp), in50 / len(exp)
        recall12_sum += r12
        recall50_sum += r50

        opp = 0
        for s in exp:
            r = rank_of.get(s)
            if r is None:
                sec_cls["miss"] += 1
            elif r <= TOP_K:
                sec_cls["already_top12"] += 1
            else:
                sec_cls["rerank_opportunity"] += 1
                opp += 1
        if opp:
            rescuable_queries.append((q, opp, [(s, rank_of.get(s)) for s in exp]))

        flag = "  <-- rescuable" if opp else ""
        print(f"{q.get('scenario',''):<18}{r12:>6.2f}{r50:>6.2f}  {q['query'][:40]}{flag}")

    n = len(QUERIES)
    total_sec = sum(sec_cls.values())
    print(f"\nSUMMARY")
    print(f"  Section Recall@12: {recall12_sum / n:.3f}")
    print(f"  Section Recall@50: {recall50_sum / n:.3f}")
    print(f"  gap (rerank ceiling): {(recall50_sum - recall12_sum) / n:.3f}\n")
    print(f"  expected sections: {total_sec}")
    print(f"    already_top12:      {sec_cls['already_top12']}")
    print(f"    rerank_opportunity: {sec_cls['rerank_opportunity']}  (rank 13-50)")
    print(f"    miss (>50):         {sec_cls['miss']}")

    if rescuable_queries:
        print(f"\nRESCUABLE TARGETS (rank 13-50 — reranking could lift these):")
        for q, opp, ranks in rescuable_queries:
            print(f"  • [{q.get('scenario')}] {q['query'][:48]}")
            for s, r in ranks:
                tag = "TOP12" if (r and r <= TOP_K) else ("RESCUE@" + str(r) if r else "MISS")
                print(f"       {s:<26} {tag}")

    print("\nVERDICT")
    opp = sec_cls["rerank_opportunity"]
    if opp == 0:
        print("  No expected section sits in 13-50. Reranking would rescue NOTHING on this")
        print("  set — the gap is misses (>50), which need better retrieval, not reranking.")
    else:
        print(f"  {opp} expected section(s) sit in 13-50 → a cross-encoder reranker could lift")
        print(f"  them into the LLM's view. Reranking is justified; ceiling = +{(recall50_sum-recall12_sum)/n:.2f} recall.")


if __name__ == "__main__":
    main()
