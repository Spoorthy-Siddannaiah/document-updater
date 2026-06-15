"""Chunking strategy experiment — header-based vs fixed-size.

Question: is header-based chunking actually better than fixed-size windows, or did
we just assume it? This measures it. For each strategy we build a fresh index
(embed + BM25), run the SAME hybrid retrieval, and report File Hit@5 over the
retrieval queries.

We compare at FILE level (not section) because fixed-size chunks have different ids
than the section labels — file membership is the only fair cross-strategy metric.

Run:
    cd backend && python -m eval.chunking_experiment
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from app import config
from app.chunking import Chunk, load_and_chunk
from app.retrieval import BM25, index, reciprocal_rank_fusion

DOCS = config.DOCS_ROOT
HIT_K = 5
QUERIES = [q for q in json.loads((Path(__file__).parent / "queries.json").read_text())
           if q["expected_files"]]


def fixed_chunks(size: int, overlap: int) -> list[Chunk]:
    """Sliding character window of `size` with `overlap`, per file."""
    out: list[Chunk] = []
    step = size - overlap
    for p in sorted(DOCS.rglob("*.md")):
        rel = p.relative_to(DOCS).as_posix()
        text = p.read_text(encoding="utf-8")
        title = p.stem.replace("_", " ").title()
        i = idx = 0
        while i < len(text):
            piece = text[i : i + size]
            if piece.strip():
                start_line = text.count("\n", 0, i) + 1
                out.append(Chunk(
                    chunk_id=f"{rel}::fix{idx}", file=rel, doc_title=title,
                    heading=title, header_path=[title], text=piece,
                    start_line=start_line, end_line=start_line,
                ))
                idx += 1
            i += step
    return out


def embed(chunks: list[Chunk]) -> np.ndarray:
    inputs = [f"{c.file} | {c.breadcrumb}\n\n{c.text}" for c in chunks]
    vecs: list[list[float]] = []
    for i in range(0, len(inputs), 128):
        resp = index.client.embeddings.create(model=config.EMBED_MODEL, input=inputs[i : i + 128])
        vecs.extend(d.embedding for d in resp.data)
    m = np.asarray(vecs, dtype=np.float32)
    m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-9
    return m


def evaluate(chunks: list[Chunk]) -> float:
    by_id = {c.chunk_id: c for c in chunks}
    matrix = embed(chunks)
    bm = BM25()
    bm.fit(chunks)
    hits = []
    for q in QUERIES:
        qv = index._embed_query(q["query"])
        scores = matrix @ qv
        dense = [chunks[i].chunk_id for i in np.argsort(-scores)[: config.DENSE_DEPTH]]
        bm25 = [cid for cid, _ in bm.search(q["query"], config.BM25_DEPTH)]
        fused = [cid for cid, _ in reciprocal_rank_fusion([dense, bm25])][:HIT_K]
        files = {by_id[c].file for c in fused if c in by_id}
        hits.append(1.0 if set(q["expected_files"]) & files else 0.0)
    return sum(hits) / len(hits)


def main() -> None:
    strategies = {
        "header-based (current)": load_and_chunk(DOCS),
        "fixed 250 chars":  fixed_chunks(250, 50),
        "fixed 1000 chars": fixed_chunks(1000, 150),
        "fixed 2000 chars": fixed_chunks(2000, 200),
    }
    print(f"{len(QUERIES)} queries | metric: File Hit@{HIT_K} | retrieval: dense+BM25 RRF\n")
    print(f"{'strategy':<26}{'#chunks':>9}{'File Hit@5':>12}")
    print("-" * 47)
    for name, chunks in strategies.items():
        print(f"{name:<26}{len(chunks):>9}{evaluate(chunks):>12.3f}")


if __name__ == "__main__":
    main()
