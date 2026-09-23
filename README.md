# Doc Update Assistant

Doc Update Assistant turns a natural-language product change into reviewable
documentation edits.

It is a human-in-the-loop AI assistant for keeping technical documentation in
sync with fast-moving product and API changes. A user describes what changed, the
backend retrieves the most relevant documentation sections, an LLM proposes
section-level edits, and the UI lets the user review, edit, approve, reject, and
save the results.

The repository includes the OpenAI Agents SDK docs as a sample corpus in
`backend/data/agents-sdk-docs/`, so the app can run locally without private
company documents.

```text
Change request -> retrieval -> AI suggestions -> human review -> saved edits
```

## Features

- Natural-language documentation change requests
- Markdown-aware chunking by heading, with line spans and breadcrumbs
- Hybrid retrieval with embeddings, BM25, reciprocal-rank fusion, and term matching
- Batched AI edit generation for long or cross-cutting changes
- Review UI with side-by-side diffs, confidence labels, approve/reject/edit flows
- Safe save path that writes edited copies under `backend/result/`
- Evaluation scripts for retrieval quality and suggestion quality
- Optional LangGraph-style orchestration, PostgreSQL persistence, and Langfuse tracing

## Tech Stack

- **Frontend:** Next.js App Router, React, TypeScript
- **Backend:** FastAPI, Pydantic, NumPy
- **AI:** OpenAI embeddings and chat completions
- **Retrieval:** dense vectors, BM25, RRF, exact-term matching
- **Optional production path:** LangGraph, PostgreSQL, Langfuse

## What This Demonstrates

This project is meant to show more than a basic LLM wrapper. The core problem is
turning an ambiguous product change into precise, reviewable documentation edits
without blindly rewriting files.

- **Retrieval design:** combines semantic search with BM25 and exact-term
  matching, because documentation updates often involve both natural language and
  exact API names.
- **Structured generation:** asks the model for section-level replacement edits
  with reasons and confidence, then maps those edits back to trusted local chunk
  metadata instead of relying on model-provided file locations.
- **Human-in-the-loop workflow:** treats AI output as a draft. Users can inspect
  diffs, edit suggestions, approve, reject, regenerate, and save.
- **Safety around writes:** applies edits only when the original section text
  still matches, and writes edited copies under `backend/result/` instead of
  mutating the source docs.
- **Evaluation mindset:** includes labelled eval queries for retrieval and
  suggestion quality, covering exact symbols, paraphrases, long sections,
  unrelated requests, and hallucination guards.
- **Production awareness:** documents tradeoffs around persistence, vector
  storage, pull-request writeback, auth, observability, and latency.

## Quickstart

### 1. Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Add your OpenAI API key to `backend/.env`:

```bash
OPENAI_API_KEY=your_api_key_here
```

Start the API:

```bash
export $(grep -v '^#' .env | xargs)
uvicorn app.main:app --reload --port 8000
```

The first run builds embeddings for the sample docs. Embeddings are cached under
`backend/.cache/embeddings/`, so later runs are faster.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

Example request:

```text
WebSearchTool now requires a mandatory api_key argument when constructed.
```

## How It Works

1. The user enters a product or API change.
2. Markdown docs are chunked by heading-aware sections.
3. The backend retrieves likely affected chunks using dense search, BM25, RRF,
   and exact-term matching.
4. Candidate sections are sent to the model in small concurrent batches.
5. The model returns structured replacement suggestions.
6. The backend reattaches trusted local metadata such as file path and line span.
7. The UI shows diffs and lets the user approve, reject, edit, or regenerate.
8. Approved edits are applied to an in-memory overlay and copied to
   `backend/result/`.

The source files in `backend/data/agents-sdk-docs/` are not modified by the save
flow.

## Architecture

```text
frontend/
  Next.js review UI
    |
    v
backend/
  FastAPI routes
    |
    +-- chunking.py      Markdown section splitting
    +-- retrieval.py     embeddings, BM25, RRF, term matching
    +-- suggest.py       LLM suggestion generation
    +-- store.py         review sessions and save overlay
    +-- agentic_workflow.py
                         optional orchestration path
```

Default retrieval settings live in `backend/app/config.py`:

```python
TOP_K = 12
DENSE_DEPTH = 20
BM25_DEPTH = 20
SUGGEST_BATCH = 4
MAX_PARALLEL_BATCHES = 6
```

## Evaluation

Evaluation scripts live in `backend/eval`.

Run retrieval evals:

```bash
cd backend
python3 -m eval.run_eval
```

Run suggestion-quality evals:

```bash
cd backend
python3 -m eval.run_suggestion_eval
```

The eval suite covers:

- exact API and signature changes
- renames and deprecations
- code-block edits
- natural-language paraphrases
- long-section retrieval
- unrelated no-op requests
- hallucination guards

Current retrieval snapshot for the hybrid pipeline:

| Metric            | Value |
| ----------------- | ----: |
| File Hit@5        |  0.95 |
| Section Hit@5     |  0.82 |
| Section Recall@12 |  0.82 |
| Section Recall@50 |  0.97 |
| MRR@5             |  0.54 |

## API Overview

| Method | Path                                              | Purpose                                    |
| ------ | ------------------------------------------------- | ------------------------------------------ |
| `GET`  | `/api/health`                                     | Health check and chunk count               |
| `POST` | `/api/query`                                      | Generate suggestions for a change request  |
| `GET`  | `/api/sessions/{session_id}`                      | Read a review session                      |
| `POST` | `/api/sessions/{sid}/suggestions/{id}/approve`    | Approve a suggestion                       |
| `POST` | `/api/sessions/{sid}/suggestions/{id}/reject`     | Reject a suggestion                        |
| `PUT`  | `/api/sessions/{sid}/suggestions/{id}`            | Edit suggestion text                       |
| `POST` | `/api/sessions/{sid}/suggestions/{id}/regenerate` | Regenerate one suggestion                  |
| `POST` | `/api/sessions/{sid}/save`                        | Save approved edits to the result overlay  |
| `GET`  | `/api/documents/{file}`                           | Read a document with overlay edits applied |

Optional orchestration endpoint:

| Method | Path                        | Purpose                                      |
| ------ | --------------------------- | -------------------------------------------- |
| `POST` | `/api/suggestions/generate` | Run the agentic workflow with tracing fields |

## Project Layout

```text
backend/
  app/
    main.py              FastAPI app and routes
    chunking.py          Markdown chunking
    retrieval.py         Retrieval index and ranking
    suggest.py           LLM edit generation
    store.py             Review sessions and save overlay
    models.py            Pydantic schemas
    agentic_workflow.py  Optional orchestrated workflow
  data/agents-sdk-docs/  Sample documentation corpus
  eval/                  Retrieval and suggestion evals

frontend/
  app/page.tsx           Main review UI
  components/            Suggestion cards and diff views
  lib/                   API client and diff helpers
```

## Design Tradeoffs

- Review state is in memory for local simplicity. A production version would use
  durable per-user sessions.
- Approved edits are written to `backend/result/`. A production version would
  open a branch and pull request against the docs repo.
- Vectors are stored in process memory because the sample corpus is small. A
  larger corpus would use pgvector, Qdrant, or a similar vector store.
- The model returns full-section replacements because they are easier to validate
  and review than fragile patch snippets.
- Long or cross-cutting updates are batched concurrently to balance coverage and
  latency.

## Future Improvements

- Add authenticated multi-user sessions
- Write approved edits to a Git branch and open pull requests
- Stream suggestion progress for long-running requests
- Add Markdown validation for generated edits
- Add CI gates for retrieval and suggestion evals
- Add token, cost, and latency dashboards
- Add reranking for cases where the right section appears below the top results
