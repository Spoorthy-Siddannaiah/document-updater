# Agentic Orchestration And Observability

## Why LangGraph Is Useful Here

LangGraph turns suggestion generation into an explicit state machine instead of
one large function. That matters because this workflow has multiple failure
points: retrieval can fail, model JSON can be malformed, validation can reject
edits, diff generation can expose no-op changes, and persistence can fail. Each
node is small enough to unit test and trace. The graph also makes fallback paths
visible: retrieval or validation failures still produce a persisted run record
instead of disappearing inside logs.

## Workflow Nodes

1. `load_request_context` validates the request, creates the Langfuse trace, and
   stores run metadata such as `run_id`, `user_id`, `tenant_id`, and
   `document_id`.
2. `retrieve_relevant_chunks` runs hybrid retrieval and returns trusted chunk
   metadata: `chunk_id`, `file_path`, `section`, `text`, and `score`.
3. `generate_edit_suggestions` calls the LLM with candidate chunks and asks for
   structured JSON. The LLM only proposes text; it never writes files.
4. `validate_suggestions` parses the raw model output with Pydantic, rejects
   unknown chunks, repairs missing source metadata from the retrieval result, and
   drops no-op edits.
5. `generate_side_by_side_diff` compares original chunk text with proposed text
   using a line-level side-by-side diff.
6. `persist_suggestions` writes the run, validated suggestions, retrieved chunk
   ids, prompt hash, model, latency, status, validation errors, and Langfuse
   trace id to PostgreSQL when `DATABASE_URL` is configured.
7. `return_result` finalizes the response for the API layer and updates Langfuse
   with the final status.

## State Passed Between Nodes

The graph state is `SuggestionGenerationState`, a `TypedDict`. It carries request
context, model configuration, prompt hash, trace id, retrieved chunks, raw model
edits, validated suggestions, diffs, validation errors, token usage, node names,
latency, status, and persistence outcome. The important design choice is that
trusted source metadata comes from retrieval, not from the LLM.

## PostgreSQL Vs Langfuse

PostgreSQL stores durable product data: suggestion runs, edit suggestions, eval
runs, eval results, statuses, diffs, prompt hashes, retrieved chunk ids, and
reviewable text. It is the source of truth for the application.

Langfuse stores observability data: traces, spans for every LangGraph node,
latency, token usage when available, validation errors, tool/model calls, and
final status. It is used for debugging and comparing behavior across runs, not
as the product database.

## Interview Value

This design shows that the agent is controlled, observable, and testable. You can
explain that LangGraph provides orchestration, Pydantic provides validation
guardrails, PostgreSQL provides durable state, and Langfuse provides traceability.
It also demonstrates human-in-the-loop safety: the LLM can suggest edits but
cannot directly modify files. The eval harness shows engineering maturity
because prompt and model changes can be compared on a fixed dataset instead of
judged by vibes.

## Node Failure Modes

- `load_request_context`: missing request fields, bad tenant/user context, or
  trace setup failure.
- `retrieve_relevant_chunks`: embedding API failure, empty index, bad query,
  weak recall, or hybrid retrieval returning irrelevant chunks.
- `generate_edit_suggestions`: provider timeout, rate limit, invalid JSON, high
  latency, or model hallucinating chunk ids.
- `validate_suggestions`: malformed JSON, unknown `chunk_id`, empty edit,
  invalid confidence, no-op edit, or missing required source mapping.
- `generate_side_by_side_diff`: very large chunks causing expensive diffs or
  proposed text that is too different to review comfortably.
- `persist_suggestions`: database outage, schema mismatch, bad JSON payload, or
  duplicate ids.
- `return_result`: serialization issues or incomplete final metadata.

## Interview Q&A

**LangGraph:** Use it when an AI workflow has state, branching, retries, or
observability needs. Here it makes each step visible and testable.

**LangChain:** Use it selectively for model calls, retrievers, prompts, and
loaders when it reduces code. This project keeps retrieval and validation in
plain Python because those parts are clearer locally.

**Langfuse:** Use it for traces, spans, model metadata, token usage, validation
errors, and latency. It answers "what happened during this run?".

**Evals:** Use a fixed dataset to compare prompt/model versions. Track file
match, chunk match, diff similarity, precision, recall, hallucinations,
validation failures, latency, and cost.

**Safety:** The model output is treated as untrusted. It must pass Pydantic
validation and human review before any file change is accepted.
