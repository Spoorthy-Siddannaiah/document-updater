"""Retrieval eval harness.

Reports, per retrieval variant, three simple "did we retrieve the right thing in
the top-5" hit rates over the labelled queries in queries.json:

  File Hit@5     — is the right FILE among the top-5 chunks?
  Section Hit@5  — is the right SECTION among the top-5 chunks?   (stricter)
  Keyword Hit@5  — does a top-5 chunk literally contain the code symbol the
                   query names (e.g. `as_tool`)?  Symbol auto-extracted from the
                   query, so no extra labels. Skipped for queries with no symbol.

@5 (not @12) because the suggestion model should see the right thing near the top,
not buried. Variants needing embeddings are skipped if OPENAI_API_KEY is unset.

Run:
    cd backend && python -m eval.run_eval
"""

from __future__ import annotations

import json
from pathlib import Path

from app import config
from app.chunking import load_and_chunk
from app.retrieval import index, reciprocal_rank_fusion, extract_symbols

K = config.TOP_K        # pipeline depth fed to the LLM
HIT_K = 5               # cutoff for the Hit@5 metrics
ALL_QUERIES = json.loads((Path(__file__).parent / "queries.json").read_text())
QUERIES = [q for q in ALL_QUERIES if q["expected_files"]]
NOOP_QUERIES = [q for q in ALL_QUERIES if not q["expected_files"]]
HAS_KEY = bool(config.OPENAI_API_KEY)


def setup() -> None:
    if HAS_KEY:
        index.build()
    else:
        index.chunks = load_and_chunk(config.DOCS_ROOT)
        index.by_id = {c.chunk_id: c for c in index.chunks}
        index.bm25.fit(index.chunks)


# ---- retrieval variants: each returns a ranked list of chunk_ids -------------
def v_dense(q: str) -> list[str]:
    return [c.chunk_id for c, _ in index.search(q, K)]


def v_bm25(q: str) -> list[str]:
    return [cid for cid, _ in index.bm25.search(q, K)]


def v_symbol(q: str) -> list[str]:
    return [c.chunk_id for c in index.symbol_search(q)][:K]


def v_dense_bm25(q: str) -> list[str]:
    dense = [c.chunk_id for c, _ in index.search(q, config.DENSE_DEPTH)]
    bm25 = [cid for cid, _ in index.bm25.search(q, config.BM25_DEPTH)]
    return [cid for cid, _ in reciprocal_rank_fusion([dense, bm25])][:K]


def _with_symbol(base: list[str], q: str) -> list[str]:
    out, seen = list(base), set(base)
    for c in index.symbol_search(q):
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            out.append(c.chunk_id)
    return out


def v_dense_symbol(q: str) -> list[str]:
    return _with_symbol(v_dense(q), q)


def v_full(q: str) -> list[str]:
    return _with_symbol(v_dense_bm25(q), q)


def v_bm25_symbol(q: str) -> list[str]:
    return _with_symbol(v_bm25(q), q)


VARIANTS = {
    "dense": (v_dense, True),
    "bm25": (v_bm25, False),
    "symbol": (v_symbol, False),
    "dense+bm25 (RRF)": (v_dense_bm25, True),
    "dense+bm25+symbol (FULL)": (v_full, True),
}


def files_of(chunk_ids: list[str]) -> set[str]:
    return {index.by_id[c].file for c in chunk_ids if c in index.by_id}


def evaluate(fn) -> dict:
    file_hits: list[float] = []
    sec_hits: list[float] = []
    kw_hits: list[float] = []
    for item in QUERIES:
        ids = fn(item["query"])[:HIT_K]
        got_files = files_of(ids)
        got_sections = set(ids)

        expected = set(item["expected_files"])
        file_hits.append(1.0 if expected & got_files else 0.0)

        sections = set(item.get("expected_sections", []))
        if sections:
            sec_hits.append(1.0 if sections & got_sections else 0.0)

        symbols = extract_symbols(item["query"])
        if symbols:
            text = " ".join(index.by_id[c].text for c in ids if c in index.by_id)
            kw_hits.append(1.0 if any(s in text for s in symbols) else 0.0)

    return {
        "file": sum(file_hits) / len(file_hits),
        "sec": sum(sec_hits) / len(sec_hits) if sec_hits else None,
        "kw": sum(kw_hits) / len(kw_hits) if kw_hits else None,
        "n_sec": len(sec_hits),
        "n_kw": len(kw_hits),
    }


def main() -> None:
    setup()
    n_sec = sum(1 for q in QUERIES if q.get("expected_sections"))
    n_kw = sum(1 for q in QUERIES if extract_symbols(q["query"]))
    print(f"Corpus: {len(index.chunks)} chunks | retrieval queries: {len(QUERIES)} "
          f"(+{len(NOOP_QUERIES)} no-op) | Hit@{HIT_K}")
    print(f"  Section Hit over {n_sec} section-labelled | Keyword Hit over {n_kw} symbol-bearing")
    print(f"OPENAI_API_KEY: {'set' if HAS_KEY else 'UNSET — dense variants skipped'}\n")

    fmt = lambda x: f"{x:.3f}" if x is not None else "  -  "
    print(f"{'variant':<28} {'File Hit@5':>11} {'Section Hit@5':>14} {'Keyword Hit@5':>14}")
    print("-" * 70)
    for name, (fn, needs_key) in VARIANTS.items():
        if needs_key and not HAS_KEY:
            print(f"{name:<28} {'(skipped)':>11}")
            continue
        m = evaluate(fn)
        print(f"{name:<28} {fmt(m['file']):>11} {fmt(m['sec']):>14} {fmt(m['kw']):>14}")


if __name__ == "__main__":
    main()
