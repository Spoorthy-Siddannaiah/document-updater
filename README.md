# Pluno Doc Updater

Pluno Doc Updater turns a natural-language product change into reviewable
documentation edits. The demo corpus is the OpenAI Agents SDK documentation in
`backend/data/agents-sdk-docs/`.

```text
User query -> retrieve affected sections -> generate edit suggestions -> review -> save
```

The app is intentionally human-in-the-loop: the model proposes edits, and the
user reviews, edits, approves, or rejects them before saving.

## Stack

- **Backend:** FastAPI, Markdown chunking, hybrid retrieval, OpenAI suggestions,
  in-memory review sessions.
- **Frontend:** Next.js App Router, TypeScript, suggestion review UI, line diff,
  approve/reject/edit controls.
- **AI:** `text-embedding-3-small` for retrieval embeddings and `gpt-4o` for edit
  suggestions.

## Run

Run the backend and frontend in separate terminals. The backend needs an OpenAI
API key because retrieval embeddings and edit suggestions call OpenAI models.

### Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# add OPENAI_API_KEY to .env
export $(grep -v '^#' .env | xargs)
uvicorn app.main:app --reload --port 8000
```

The first run embeds the documentation chunks. Embeddings are cached under
`backend/.cache/embeddings/`, keyed by embedding model and chunk text hash, so
unchanged chunks are reused on later runs.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`. In development, the Next.js app proxies `/api/*`
requests to the FastAPI backend on `http://localhost:8000`.

Example query:

```text
We don't support agents as_tool anymore, other agents should only be invoked via handoff.
```

## Outputs And Logs

- **Edited documents:** approved edits are stored in an in-memory overlay. They
  can be viewed through the UI or `GET /api/documents/{file}`. A copy of each
  saved edited file is also written under `backend/result/`, preserving the
  original relative path. The source Markdown files under
  `backend/data/agents-sdk-docs/` are not changed.
- **Embedding cache:** chunk embeddings are stored under
  `backend/.cache/embeddings/`.
- **Logs:** backend timing and request logs are written to stdout and
  `backend/log/rag.log`. The eval scripts print their reports to stdout; saved
  snapshots such as `backend/eval/report_results.txt` and
  `backend/eval/judge_results.txt` capture representative runs.

After saving an edit, verify the written copy and logs with:

```bash
rg -n "max_num_results=(3|10)" backend/data/agents-sdk-docs/tools.md backend/result/tools.md
tail -n 30 backend/log/rag.log
```

## Product Flow

1. The user enters a documentation change request.
2. The backend retrieves likely affected documentation chunks.
3. The suggestion model receives only those candidate sections.
4. The model returns structured section-level replacement suggestions.
5. The UI shows diffs with approve, reject, and edit controls.
6. Saving applies approved edits to an in-memory document overlay and writes
   edited file copies under `backend/result/`.

Original files in `backend/data/agents-sdk-docs/` are not mutated by the demo save
path.

## AI Pipeline

### Chunking

Markdown files are split in [backend/app/chunking.py](backend/app/chunking.py).

- Chunks are based on Markdown headings, which match documentation edit
  boundaries better than fixed-size windows.
- Code fences are tracked, so `#` inside a code block is not treated as a heading.
- Each chunk stores `chunk_id`, file path, document title, heading, heading
  breadcrumb, text, and 1-based line span.
- Sections over `MAX_CHARS = 4000` are hard-split with `200` characters of
  overlap.

Current corpus shape:

| Item                              | Count           |
| --------------------------------- | --------------: |
| Markdown files                    |              36 |
| Chunks                            |             440 |
| Average chunk size                | ~939 characters |
| Final chunks exactly 4000 chars   |               6 |
| Original sections hard-split      |               5 |

The chunk metadata is used for retrieval context, UI display, and safe mapping
back to the original document location.

### Retrieval

Retrieval is implemented in [backend/app/retrieval.py](backend/app/retrieval.py)
and used by [backend/app/suggest.py](backend/app/suggest.py).

The current retrieval stack:

1. **Dense retrieval:** embed the query, compare against all chunk embeddings with
   cosine similarity.
2. **BM25 retrieval:** keyword retrieval over chunk text for exact API names,
   config values, code identifiers, and numbers.
3. **RRF fusion:** merge dense and BM25 ranked lists with Reciprocal Rank Fusion.
4. **Exact-term expansion:** add every chunk containing extracted quoted terms or
   code-like symbols from the query.

Default retrieval settings are in [backend/app/config.py](backend/app/config.py):

```python
TOP_K = 12
DENSE_DEPTH = 20
BM25_DEPTH = 20
SUGGEST_BATCH = 4
```

Dense retrieval helps when the query uses different wording from the docs. BM25
helps when the query contains exact terms like `WebSearchTool`, `Agent.as_tool`,
`max_turns=None`, or numeric defaults. RRF combines the two without comparing
incompatible raw scores.

The suggestion step sends candidates to the model in batches of four sections.
For example, 12 candidate chunks become three LLM calls, each with the same user
query and a different subset of sections. Multiple batches run concurrently to
keep latency closer to one model call while keeping each prompt small.

### Suggestion Generation

Suggestion generation is implemented in [backend/app/suggest.py](backend/app/suggest.py).

The model receives candidate sections rendered with:

- `chunk_id`
- file path
- breadcrumb/location
- full chunk text

The model returns JSON edits with:

- `chunk_id`
- full replacement section text
- reason
- confidence

The backend reattaches original text and line spans from local state. It does not
trust the model for source locations.

### Review And Save

Review state is stored in memory in [backend/app/store.py](backend/app/store.py).
Approved edits are applied by exact-text replacement into a per-file overlay and
written as edited copies under `backend/result/`. If an original section no
longer matches, the edit is skipped instead of being applied blindly.

## Implementation Summary

The implementation is a small RAG system specialized for documentation edits:

1. Markdown documents are chunked by heading, with metadata for file path,
   breadcrumb, heading, and line span.
2. Chunk embeddings are generated with OpenAI and cached locally by embedding
   model plus chunk-text hash.
3. A user query is retrieved against the 440 chunks using dense search, BM25, RRF,
   and exact-term expansion.
4. Candidate sections are inserted into a fixed suggestion prompt and sent to the
   LLM in small concurrent batches.
5. The model returns structured JSON suggestions, and the backend maps them back
   to trusted local chunk metadata.
6. The UI lets the user review, edit, approve, reject, and save suggestions.
7. Saved edits are written to the in-memory overlay and copied under
   `backend/result/`; request timing is logged to `backend/log/rag.log`.

Optimizations implemented in this version:

- **Embedding reuse:** unchanged chunk embeddings are reused across restarts.
- **Hybrid retrieval:** dense search handles paraphrases, while BM25 and exact
  terms handle API names, numbers, and code-like symbols.
- **Completeness union:** exact-term matches are added beyond the top-12 ranked
  chunks for rename-everywhere and symbol-heavy changes.
- **Concurrent suggestion batches:** multiple small LLM prompts run in parallel
  to reduce latency without sending one very large prompt.
- **Defensive save path:** edits are applied only when the original section text
  still matches, avoiding blind writes to stale content.



## Evaluation

The evals live in [backend/eval](backend/eval).

### Retrieval Eval

[backend/eval/queries.json](backend/eval/queries.json) contains 22 retrieval
queries. The set covers straightforward product changes, API/signature changes,
renames and deprecations, paraphrases, hard-split long sections, and unrelated
no-op requests.

The base cases are grounded in changelog-style Agents SDK changes, where a
product/API update implies one or more documentation updates. Additional
scenario cases were added to exercise failure modes that changelog examples do
not cover well: vague user wording, long sections, cross-cutting updates,
unrelated requests, and hallucination guards.

Run:

```bash
cd backend
python3 -m eval.run_eval
```

Useful retrieval metrics:

- **File Recall@K:** expected files represented in retrieved chunks.
- **Section Recall@K:** expected sections represented in retrieved chunks.
- **MRR / first correct rank:** how early the first correct result appears.
- **Candidate count:** how much context is sent to the suggestion model.

Current results for the production pipeline (dense + BM25 fused with RRF, plus the
term-match completeness union), measured at K=12:

| Metric             | Value | Reading                                                |
| ------------------ | ----: | ------------------------------------------------------ |
| File Hit@5         |  0.95 | The correct file is in the top 5                       |
| Section Hit@5      |  0.82 | At least one correct section is in the top 5           |
| Section Recall@12  |  0.82 | Share of expected sections reaching the model          |
| Section Recall@50  |  0.97 | Wider-pool recall; the gap is the reranking ceiling    |
| MRR@5              |  0.54 | The correct section is often found but ranked mid-pack |
| Avg candidates     |   ~24 | Top-12 ranked chunks plus the uncapped term-match union |

Why the hybrid, measured per retriever (Section Recall on the labelled set):

| Retriever          | Strength                         | Weakness                                      |
| ------------------ | -------------------------------- | --------------------------------------------- |
| Dense only         | Paraphrases and related wording  | Misses exact numbers (0.50 on number changes) |
| BM25 only          | Exact symbols, numbers, paths    | Misses pure paraphrases (0.67)                |
| Dense + BM25 (RRF) | Recovers the misses of each      | Best overall; paraphrase gap remains          |

Completeness on cross-cutting changes is measured separately
([backend/eval/completeness.py](backend/eval/completeness.py)). Renaming a term
that appears in 76 places reaches only about 14% recall from the ranked top-12,
but about 93% once the uncapped term-match union is added, which is why that union
exists.

The hard-split cases currently include targets such as:

```text
tools.md::769.0
sandbox/guide.md::691.0
guardrails.md::68.0
sandbox/clients.md::81.0 / sandbox/clients.md::81.1
examples.md::5.1 / examples.md::5.2
```

A lightweight BM25-only check found the hard-split targets in the top 5 for the
current targeted cases. The natural paraphrase for the Codex output-limit change
still found `tools.md::769.0`, but lower in the ranking, which is the kind of case
where dense retrieval and reranking matter.

### Suggestion Eval

[backend/eval/suggestion_queries.json](backend/eval/suggestion_queries.json)
contains 23 suggestion-quality queries. It checks more than simple successful
edits: deprecations/removals, cross-cutting renames, code-block edits,
hard-split updates, vague requests, unrelated no-op requests, and hallucination
guards.

This mirrors the retrieval eval split: realistic changelog-style changes provide
the main signal, while targeted scenario cases test whether the system avoids
over-editing, missing long-section updates, or inventing unsupported replacements.

Run:

```bash
cd backend
python3 -m eval.run_suggestion_eval
```

The suggestion eval performs deterministic checks:

- edit count within expected bounds
- edit landed in the expected file family
- required terms are present
- stale terms are removed
- unsupported/forbidden terms are absent
- forbidden files are not edited
- original text exists in the source document

These checks approximate two important quality dimensions:

- **Completeness:** did the system cover the expected places to update?
- **Faithfulness:** did the edit stay grounded in the user request and source
  sections without touching unrelated docs?

Faithfulness is also scored by a separate model acting as a judge
([backend/eval/judge.py](backend/eval/judge.py)), since whether every edit is
justified cannot be checked by string matching. Completeness is read from the
labelled target sections, because a judge only sees the edits that were made and
cannot detect a place that was silently missed.

Current results by query type (file-level suggestion recall, and faithfulness from
the judge):

| Query type            |  n | Suggestion recall (file) | Faithfulness             |
| --------------------- | -: | -----------------------: | ------------------------ |
| Exact-symbol          | 10 |                     1.00 | 0.88                     |
| Cross-cutting rename  |  2 |                     1.00 | 1.00                     |
| Code-block edit       |  1 |                     1.00 | 1.00                     |
| Number / value        |  2 |                     1.00 | 0.50                     |
| Natural paraphrase    |  3 |                     0.67 | 0.83                     |
| No-op / vague         |  5 |                      n/a | 2 of 5 correctly silent  |

Reading the results:

- Once retrieval surfaces the right file, the model edits it reliably, so quality
  is retrieval-bound rather than generation-bound.
- The weakest cases are natural-language paraphrases without an explicit symbol,
  and vague requests where the model edits more than it should.
- Number changes are a known gap: the model often describes the change in prose
  but does not apply the new value in the example, which the judge penalizes.
- Numbers are small (around 20–27 queries), so the values are directional and best
  used to compare configurations rather than as absolute grades.

## Tradeoffs

The current implementation favors a clear, inspectable demo over production
infrastructure.

- **State:** sessions and suggestions are in memory. Production would use
  Postgres tables for sessions, suggestions, approvals, and audit logs.
- **Save path:** approved edits are applied to a per-file overlay and copied to
  `backend/result/`. Production would write a branch, commit, and pull request.
- **Retrieval storage:** vectors live in a NumPy matrix in process memory.
  Production would use pgvector or Qdrant with persistent chunk metadata.
- **Chunk storage:** docs are re-chunked at startup. Production would store
  chunks, hashes, line spans, and doc versions.
- **Embedding refresh:** cached by model and chunk text hash. Production would
  use incremental re-indexing with config-aware cache keys.
- **Suggestion generation:** synchronous request path. Production would move
  generation to an async job queue with progress updates or streaming.
- **LLM output:** JSON mode and defensive parsing. Production would use strict
  structured outputs with schema validation.
- **Edit granularity:** full-section replacement. Production would consider
  span-level patches or validated unified diffs.
- **Auth and concurrency:** no auth and best-effort exact-text replacement.
  Production would add auth/RBAC, per-user sessions, and conflict resolution.
- **Observability and evals:** basic logs and labelled scripts. Production would
  add tracing, token/cost metrics, p50/p95 latency, CI eval gates, and production
  acceptance metrics.

### Latency Profile

Each request is timed by stage (logged in `app/suggest.py` and `app/main.py`).
The breakdown drives where optimization effort is spent.

- **Chunking/index build:** one-time at startup. Chunk embeddings are cached by
  model and content hash, so restarts are near-instant when docs are unchanged.
- **Retrieval:** roughly sub-second to a few seconds, including query embedding,
  cosine search, BM25, and term matching. The in-memory search over 440 chunks is
  not the bottleneck.
- **LLM suggestion generation:** roughly 6-30 seconds and usually around 90% of
  request time. This is output-token and model-latency bound.

The LLM stage is the bottleneck, so that is where the speed decisions are made.
A cross-cutting change can produce several review batches; running them
sequentially exceeded a proxy timeout, so batches are issued concurrently with a
small batch size (`SUGGEST_BATCH=4`), which brought the heaviest query from about
45 s to about 20 s. This is a deliberate latency-for-completeness balance: the
uncapped term-match union guarantees coverage on rename-everywhere queries but
adds candidates and therefore batches.

With more time, the LLM stage would be reduced by routing simple changes to a
smaller, faster model, streaming partial results to the UI, caching by query, and
moving generation onto an async job queue so the request returns immediately and
suggestions stream in. Retrieval would stay as-is until corpus size makes it
matter.

## Alternatives Considered

### Whole-Corpus Prompting

The corpus is small enough to almost fit into a large context window, but sending
the entire corpus for every query increases cost, dilutes attention, and does not
scale. Retrieval keeps the prompt focused and mirrors the production shape.

### Fixed-Size Chunking

Fixed-size chunks are simple, but they cut across Markdown structure. Heading-based
chunks preserve section boundaries, breadcrumbs, and line spans, which are needed
for reviewable documentation edits.

### Vector Database From The Start

The current corpus has 440 chunks, so exact NumPy cosine search is fast and
dependency-light. A vector database becomes useful when the corpus grows, when
metadata filtering is needed, or when multiple repos/users share the system.

### LangChain Retrievers

LangChain can speed up prototyping, especially for parent-document retrieval or
vector-store integrations. This app needs custom Markdown boundaries, stable chunk
IDs, and exact line spans for editing, so the core retrieval path stays small and
explicit.

### Parent-Child Retrieval

Parent-child retrieval searches smaller child chunks and sends larger parent
sections to the model. It is useful when long sections dilute retrieval signals.
The current long-section evals still retrieve the expected chunks, so this remains
a targeted future improvement rather than first-version complexity.

### Multi-Query Expansion

Multi-query expansion helps when user wording differs strongly from docs wording.
It adds cost and another model dependency. The current system keeps single-query
retrieval and uses dense search for paraphrase coverage.

### Cross-Encoder Reranking

Reranking is useful when the expected section appears in a wider candidate pool
but falls below the top sections sent to the LLM. The right experiment is to
compare Section Recall@12 against Section Recall@50 and count how many targets
could be rescued by reranking.

### Patch Output From The Model

Patch output can produce smaller diffs, but malformed patches are common. Full
section replacement is more robust for a review UI and easier to validate.

## Production Architecture

```text
Frontend (Next.js)
  -> FastAPI API
      -> Postgres: docs, chunks, sessions, suggestions, approvals, audit log
      -> Vector DB: embeddings and metadata
      -> Object store: raw docs and snapshots
      -> Git provider: branch + PR write-back
      -> Job queue: async retrieval/suggestion workers
      -> LLM provider: embeddings, suggestions, optional reranking
```

Key production behaviors:

- re-index only changed chunks after docs change
- store chunk hashes and embedding config versions
- route approved edits through pull requests
- stream suggestion progress for long jobs
- track token usage, cost, latency, and acceptance rate
- gate retrieval/prompt changes with evals in CI

## Future Improvements

Priority improvements:

1. Expand section-level labels and add the eval report to CI so retrieval/prompt
   changes cannot silently regress quality.
2. Add token usage and estimated cost logging from model responses.
3. Add Markdown validation for generated edits, including balanced code fences and
   preserved headings.
4. Add parent-child retrieval for long sections if long-section recall drops.
5. Add cross-encoder reranking if targets appear in top 50 but not top 12.
6. Add multi-query expansion for paraphrase-heavy requests.
7. Add a consistency pass that searches for contradictions after proposed edits.
8. Detect stale suggestions and concurrent edits with stored doc versions/chunk
   hashes, then regenerate or show a conflict UI when the source section changed.
9. Add a rate-limit-aware LLM request manager that queues requests, caps
   concurrency, tracks requests/tokens per minute, and retries on `429` responses.
10. Move state and write-back to durable storage and pull requests.

## Project Layout

```text
backend/
  app/
    chunking.py     Markdown chunking with headings, code-fence awareness, line spans
    retrieval.py    Embedding cache, dense search, BM25, RRF, exact-term matching
    suggest.py      Candidate retrieval and LLM edit suggestions
    store.py        In-memory sessions and document overlay
    models.py       Pydantic schemas
    main.py         FastAPI routes
  data/agents-sdk-docs/
    *.md            Documentation corpus
  eval/
    queries.json
    suggestion_queries.json
    run_eval.py
    run_suggestion_eval.py

frontend/
  app/page.tsx
  components/SuggestionCard.tsx
  lib/api.ts
  lib/diff.ts
```

## API

| Method | Path                                             | Purpose                                            |
| ------ | ------------------------------------------------ | -------------------------------------------------- |
| `POST` | `/api/query`                                     | Submit a change request and receive suggestions    |
| `POST` | `/api/sessions/{sid}/suggestions/{id}/approve`   | Approve a suggestion                               |
| `POST` | `/api/sessions/{sid}/suggestions/{id}/reject`    | Reject a suggestion                                |
| `PUT`  | `/api/sessions/{sid}/suggestions/{id}`           | Edit suggestion text and approve it                |
| `POST` | `/api/sessions/{sid}/save`                       | Apply approved edits to the overlay/result copy    |
| `GET`  | `/api/documents/{file}`                          | Read current content, including overlay edits      |
