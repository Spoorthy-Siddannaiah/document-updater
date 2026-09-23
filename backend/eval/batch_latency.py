"""Batch-latency experiment — all-at-once vs sequential vs concurrent.

For queries whose candidate set includes long (>=4000-char / hard-split) chunks,
compare three ways of sending candidates to the LLM:

  all_at_once  one LLM call with every candidate in a single prompt
  sequential   batches of SUGGEST_BATCH, one after another
  concurrent   the same batches, issued in parallel (the production path)

Reports wall-clock latency and edit count for each. Shows why batching+concurrency
is used: a single huge prompt is slow and risks quality/truncation; sequential
batches add up; concurrent batches keep wall-time near a single call.

Run:
    cd backend && python -m eval.batch_latency
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from app import config
from app.chunking import MAX_CHARS
from app.retrieval import index
from app.suggest import _candidate_chunks, _edits_for_batch

QUERIES = [
    "Tool guardrail callbacks must now be async functions; update the ToolGuardrailFunctionOutput examples",
    "VercelSandboxClient now supports S3Mount and R2Mount through VercelBlobMountStrategy; update the hosted sandbox platform tables",
    "We don't support agents as_tool anymore, other agents should only be invoked via handoff",
]


def is_long(chunk) -> bool:
    return "." in chunk.chunk_id.split("::")[-1] or len(chunk.text) == MAX_CHARS


def all_at_once(query, cands):
    t = time.perf_counter()
    edits = _edits_for_batch(query, cands)
    return time.perf_counter() - t, len(edits)


def sequential(query, cands, bs):
    t = time.perf_counter()
    edits = []
    for i in range(0, len(cands), bs):
        edits.extend(_edits_for_batch(query, cands[i : i + bs]))
    return time.perf_counter() - t, len(edits)


def concurrent(query, cands, bs):
    batches = [cands[i : i + bs] for i in range(0, len(cands), bs)]
    t = time.perf_counter()
    edits = []
    with ThreadPoolExecutor(max_workers=min(len(batches), 10)) as pool:
        for be in pool.map(lambda b: _edits_for_batch(query, b), batches):
            edits.extend(be)
    return time.perf_counter() - t, len(edits)


def main() -> None:
    index.build()
    bs = config.SUGGEST_BATCH
    print(f"Batch-latency experiment | SUGGEST_BATCH={bs} | model={config.CHAT_MODEL}\n")

    for q in QUERIES:
        cands = _candidate_chunks(q)
        n_long = sum(1 for c, _ in cands if is_long(c))
        n_batches = (len(cands) + bs - 1) // bs
        print("=" * 70)
        print(f"query: {q[:60]}")
        print(f"candidates: {len(cands)} ({n_long} long/>=4000-char) -> {n_batches} batches\n")

        for name, fn in [
            ("all_at_once", lambda: all_at_once(q, cands)),
            ("sequential ", lambda: sequential(q, cands, bs)),
            ("concurrent ", lambda: concurrent(q, cands, bs)),
        ]:
            dt, ne = fn()
            print(f"  {name}: {dt:6.1f}s   edits={ne}")
        print()


if __name__ == "__main__":
    main()
