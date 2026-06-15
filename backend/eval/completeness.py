"""Completeness eval — answers "how do you know you didn't miss any?"

Hit@5 only checks the *best* hit. But the product's job on a cross-cutting change
("rename X everywhere") is to find ALL the places. This measures that directly:

    Recall = |retrieved ∩ all-chunks-containing-the-term| / |all-chunks-containing-it|

Ground truth needs no manual labels: for a term rename, every chunk literally
containing the term is a place that must change (grep = oracle).

We compare two strategies on that recall:
  - RRF top-K        — ranked + capped at 12 (the normal pipeline)
  - + term-match     — RRF UNION every exact-term chunk (uncapped)

The point: a capped ranked list structurally CANNOT cover 76 occurrences in 12
slots. Uncapped exact-term match can. This is where exact matching earns its place.

Run:
    cd backend && python -m eval.completeness
"""

from __future__ import annotations

import json
from pathlib import Path

from app import config
from app.chunking import load_and_chunk
from app.retrieval import index, reciprocal_rank_fusion

QUERIES = json.loads((Path(__file__).parent / "completeness_queries.json").read_text())


def setup() -> None:
    if config.OPENAI_API_KEY:
        index.build()
    else:
        index.chunks = load_and_chunk(config.DOCS_ROOT)
        index.by_id = {c.chunk_id: c for c in index.chunks}
        index.bm25.fit(index.chunks)


def all_containing(term: str) -> set[str]:
    """The oracle: every chunk literally containing the term (case-insensitive)."""
    t = term.lower()
    return {c.chunk_id for c in index.chunks if t in c.text.lower()}


def rrf_topk(query: str) -> list[str]:
    d = [c.chunk_id for c, _ in index.search(query, config.DENSE_DEPTH)]
    b = [cid for cid, _ in index.bm25.search(query, config.BM25_DEPTH)]
    return [cid for cid, _ in reciprocal_rank_fusion([d, b])][: config.TOP_K]


def main() -> None:
    setup()
    has_key = bool(config.OPENAI_API_KEY)
    print(f"Completeness eval | {len(index.chunks)} chunks | "
          f"{'dense ON' if has_key else 'dense OFF (RRF degrades to BM25-only)'}\n")
    print(f"{'query (term)':<34}{'#places':>8}{'RRF@12 recall':>15}{'+term-match':>13}")
    print("-" * 72)

    for q in QUERIES:
        expected = all_containing(q["term"])
        n = len(expected)
        if has_key:
            capped = set(rrf_topk(q["query"]))
        else:
            capped = {cid for cid, _ in index.bm25.search(q["query"], config.TOP_K)}
        term_union = capped | {c.chunk_id for c in index.term_matches(q["query"])}

        r_capped = len(expected & capped) / n if n else 0.0
        r_union = len(expected & term_union) / n if n else 0.0
        label = f"{q['term']}"
        print(f"{label:<34}{n:>8}{r_capped:>15.2f}{r_union:>13.2f}")

    print("\nReading: RRF@12 caps at 12 candidates, so recall collapses when a term")
    print("appears in many places. Uncapped term-match recovers full coverage —")
    print("that is the answer to 'how do you know you didn't miss any?'.")


if __name__ == "__main__":
    main()
