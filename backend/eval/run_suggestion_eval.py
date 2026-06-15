"""Suggestion-quality eval — runs each labelled change request through the live
suggestion engine and applies mechanical checks.

Checks per query (no LLM judge yet — deterministic substring/file checks):
  1. count     — #edits within [expected_min_edits, expected_max_edits]
  2. location  — did an edit land in one of target_sections?
  3. content   — do the edits collectively contain every must_mention term?
  4. no-damage — no must_not_edit file touched; must_not_keep phrases gone
                 from the edited sections.

Run:
    cd backend && python -m eval.run_suggestion_eval
"""

from __future__ import annotations

import json
from pathlib import Path

from app import config
from app.chunking import load_and_chunk
from app.retrieval import index
from app.suggest import generate_suggestions

QUERIES = json.loads((Path(__file__).parent / "suggestion_queries.json").read_text())

# Verbatim source text per file, to verify a suggestion edits real existing text
# (and the LLM did not invent the "original" it claims to replace).
_SOURCE = {
    c.file: (config.DOCS_ROOT / c.file).read_text(encoding="utf-8")
    for c in load_and_chunk(config.DOCS_ROOT)
}


def current_text_exists(suggestions) -> float | None:
    """Fraction of edits whose original_text is found verbatim in the source doc.
    1.0 = no hallucinated targets. None = no edits to check."""
    if not suggestions:
        return None
    ok = sum(1 for s in suggestions
             if s.original_text.strip() and s.original_text in _SOURCE.get(s.file, ""))
    return ok / len(suggestions)


def check(item, suggestions) -> dict:
    files = {s.file for s in suggestions}
    sections = {s.chunk_id for s in suggestions}
    all_new = "\n".join(s.suggested_text for s in suggestions).lower()

    n = len(suggestions)
    lo = item.get("expected_min_edits", 0)
    hi = item.get("expected_max_edits")
    count_ok = n >= lo and (hi is None or n <= hi)

    # Location at FILE level: the model legitimately chooses which section(s) to
    # edit, so requiring an exact section id is too strict. We check it landed in
    # the right file(s) (derived from the target sections).
    target_files = {t.split("::")[0] for t in item.get("target_sections", [])}
    location_ok = (not target_files) or bool(target_files & files)

    must = [m.lower() for m in item.get("must_mention", [])]
    content_ok = all(m in all_new for m in must)

    forbid_files = set(item.get("must_not_edit", []))
    if "*" in forbid_files:
        nodamage_files = (n == 0)
    else:
        nodamage_files = not (forbid_files & files)
    not_keep = [x.lower() for x in item.get("must_not_keep", [])]
    nodamage_keep = all(x not in all_new for x in not_keep)
    nodamage_ok = nodamage_files and nodamage_keep

    exists = current_text_exists(suggestions)        # anti-hallucination
    exists_ok = exists is None or exists >= 1.0

    passed = count_ok and location_ok and content_ok and nodamage_ok and exists_ok
    return {
        "n": n, "count": count_ok, "location": location_ok,
        "content": content_ok, "nodamage": nodamage_ok,
        "exists": exists, "exists_ok": exists_ok, "pass": passed,
        "files": files, "sections": sections,
    }


def main() -> None:
    index.build()
    print(f"Suggestion eval | {len(QUERIES)} queries | model={config.CHAT_MODEL}\n")
    print(f"{'scenario':<26} {'n':>2} {'cnt':>4} {'loc':>4} {'cont':>5} {'safe':>5} {'exist':>6} {'PASS':>5}")
    print("-" * 70)

    results = []
    for item in QUERIES:
        sugs, _ = generate_suggestions(item["query"])
        r = check(item, sugs)
        results.append((item, r))
        mark = lambda b: " ok " if b else " ✗  "
        ex = "  -  " if r["exists"] is None else f"{r['exists']:.2f} "
        print(f"{item['scenario']:<26} {r['n']:>2} {mark(r['count'])} {mark(r['location'])} "
              f"{mark(r['content'])} {mark(r['nodamage'])} {ex:>6} {'PASS' if r['pass'] else 'FAIL'}")

    npass = sum(1 for _, r in results if r["pass"])
    print("-" * 60)
    print(f"PASS {npass}/{len(results)}  ({100*npass/len(results):.0f}%)")

    print("\nFailures detail:")
    for item, r in results:
        if r["pass"]:
            continue
        fails = [k for k in ("count", "location", "content", "nodamage") if not r[k]]
        print(f"  [{item['scenario']}] failed: {', '.join(fails)}")
        print(f"     query: {item['query'][:70]}")
        print(f"     edits: {r['n']} -> files {sorted(r['files'])}")
        if "location" in fails:
            print(f"     wanted sections {item.get('target_sections')}, got {sorted(r['sections'])}")
        if "content" in fails:
            print(f"     wanted mentions {item.get('must_mention')}")


if __name__ == "__main__":
    main()
