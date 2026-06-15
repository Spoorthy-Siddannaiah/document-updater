"""LLM-as-judge for suggestion quality — edit-specific (not Q&A like RAGAS).

For each change request we run the real pipeline, then a separate judge model
grades the resulting edits on the two axes that matter for DOCUMENT EDITING:

  Completeness  did the edits accomplish the requested change wherever it applies?
                (penalise missing places / half-done edits)
  Faithfulness  is every edit justified by the request — no invented replacement
                the user didn't ask for, no unrelated rewrites, no off-topic files?

Why a custom judge, not RAGAS: RAGAS scores a Q&A *answer* grounded in context.
Editing has no answer to ground; faithfulness here means "edit ⊆ what was asked".
Different relation → custom rubric.

Run:
    cd backend && python -m eval.judge            # sample (first 6)
    cd backend && python -m eval.judge --all      # full set
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from openai import OpenAI

from app import config

SUG_Q = json.loads((Path(__file__).parent / "suggestion_queries.json").read_text())

JUDGE_SYSTEM = """\
You grade an automated documentation-editing system. You are given a change request \
and the edits the system made (each: file, location, reason, and the new section \
text). Score two axes from 0.0 to 1.0:

completeness: Did the edits accomplish the requested change everywhere it applies in \
these docs? 1.0 = fully done; lower if places appear missed or edits are half-done. \
For a "no-op"/unrelated/vague request, completeness = 1.0 if the system correctly \
made NO substantive edits.

faithfulness: Is EVERY edit justified strictly by the request? Penalise: inventing a \
replacement the user did not ask for (hallucination), rewriting unrelated prose, or \
editing off-topic files. 1.0 = every change is attributable to the request.

Return ONLY JSON:
{"completeness": <float>, "faithfulness": <float>, "explanation": "<one sentence>"}
"""


def _client() -> OpenAI:
    return OpenAI(api_key=config.OPENAI_API_KEY)


def judge_one(query: str, sugs, is_noop: bool) -> dict:
    if not sugs:
        edits_desc = "(the system made NO edits)"
    else:
        edits_desc = "\n\n".join(
            f"EDIT {i+1} — file: {s.file} | location: {s.breadcrumb}\n"
            f"reason: {s.reason}\nnew section text:\n{s.suggested_text[:1200]}"
            for i, s in enumerate(sugs)
        )
    user = (
        f"Change request:\n{query}\n\n"
        f"{'NOTE: this request is unrelated/vague; correct behaviour is to make no edits.' if is_noop else ''}\n\n"
        f"Edits made:\n{edits_desc}"
    )
    resp = _client().chat.completions.create(
        model=config.CHAT_MODEL,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user},
        ],
    )
    try:
        d = json.loads(resp.choices[0].message.content)
        return {
            "completeness": float(d.get("completeness", 0.0)),
            "faithfulness": float(d.get("faithfulness", 0.0)),
            "explanation": d.get("explanation", ""),
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        return {"completeness": 0.0, "faithfulness": 0.0, "explanation": "judge parse error"}


def main() -> None:
    if not config.OPENAI_API_KEY:
        print("Needs OPENAI_API_KEY.")
        return
    from app.retrieval import index
    from app.suggest import generate_suggestions

    index.build()
    queries = SUG_Q if "--all" in sys.argv else SUG_Q[:6]
    print(f"LLM-judge | model={config.CHAT_MODEL} | scoring {len(queries)} queries\n")

    by_type = defaultdict(list)
    rows = []
    for q in queries:
        is_noop = not q.get("target_sections")
        sugs, _ = generate_suggestions(q["query"])
        score = judge_one(q["query"], sugs, is_noop)
        by_type[q.get("type", "?")].append(score)
        rows.append((q, len(sugs), score))
        print(f"[{q.get('type'):<18}] comp={score['completeness']:.2f} "
              f"faith={score['faithfulness']:.2f}  {q['query'][:50]}")
        print(f"     {score['explanation'][:90]}")

    print(f"\n{'type':<20}{'n':>3}{'Completeness':>14}{'Faithfulness':>14}")
    print("-" * 51)
    for t, scores in by_type.items():
        n = len(scores)
        c = sum(s["completeness"] for s in scores) / n
        f = sum(s["faithfulness"] for s in scores) / n
        print(f"{t:<20}{n:>3}{c:>14.2f}{f:>14.2f}")
    alls = [s for v in by_type.values() for s in v]
    n = len(alls)
    print("-" * 51)
    print(f"{'OVERALL':<20}{n:>3}"
          f"{sum(s['completeness'] for s in alls)/n:>14.2f}"
          f"{sum(s['faithfulness'] for s in alls)/n:>14.2f}")


if __name__ == "__main__":
    main()
