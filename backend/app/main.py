"""FastAPI app: query -> suggestions -> review -> save."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import config
from .models import (
    Confidence,
    EditSuggestion,
    QueryRequest,
    QueryResponse,
    SessionState,
    Suggestion,
    SuggestionStatus,
)
from .retrieval import index
from .store import store
from .suggest import generate_suggestions, regenerate_one

config.LOG_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOG_FILE, encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Building doc index from %s", config.DOCS_ROOT)
    index.build()
    yield


app = FastAPI(title="Pluno Doc Updater", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "chunks": len(index.chunks)}


@app.post("/api/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    t0 = time.perf_counter()
    try:
        suggestions, considered = generate_suggestions(req.query)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    session_id = uuid.uuid4().hex
    store.create_session(session_id, req.query, suggestions)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    logger.info("POST /api/query total=%.1fs edits=%d", elapsed_ms / 1000, len(suggestions))
    return QueryResponse(
        session_id=session_id,
        query=req.query,
        suggestions=suggestions,
        considered_chunks=considered,
        elapsed_ms=elapsed_ms,
    )


@app.get("/api/sessions/{session_id}", response_model=SessionState)
def get_session(session_id: str) -> SessionState:
    session = store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session not found")
    return session


def _require(sug: Suggestion | None) -> Suggestion:
    if not sug:
        raise HTTPException(status_code=404, detail="suggestion not found")
    return sug


@app.post("/api/sessions/{session_id}/suggestions/{suggestion_id}/approve", response_model=Suggestion)
def approve(session_id: str, suggestion_id: str) -> Suggestion:
    return _require(store.set_status(session_id, suggestion_id, SuggestionStatus.approved))


@app.post("/api/sessions/{session_id}/suggestions/{suggestion_id}/reject", response_model=Suggestion)
def reject(session_id: str, suggestion_id: str) -> Suggestion:
    return _require(store.set_status(session_id, suggestion_id, SuggestionStatus.rejected))


@app.put("/api/sessions/{session_id}/suggestions/{suggestion_id}", response_model=Suggestion)
def edit(session_id: str, suggestion_id: str, body: EditSuggestion) -> Suggestion:
    return _require(store.edit_suggestion(session_id, suggestion_id, body.suggested_text))


@app.post("/api/sessions/{session_id}/suggestions/{suggestion_id}/regenerate", response_model=Suggestion)
def regenerate(session_id: str, suggestion_id: str) -> Suggestion:
    query = store.session_query(session_id)
    current = store._find(session_id, suggestion_id)
    if query is None or current is None:
        raise HTTPException(status_code=404, detail="session or suggestion not found")
    chunk = index.by_id.get(current.chunk_id)
    if chunk is None:
        raise HTTPException(status_code=404, detail="source chunk not found")
    edit = regenerate_one(query, chunk)
    if not edit:
        raise HTTPException(status_code=422, detail="model produced no new edit; keeping current")
    try:
        confidence = Confidence(edit.get("confidence", "medium"))
    except ValueError:
        confidence = Confidence.medium
    return _require(
        store.replace_suggestion(
            session_id, suggestion_id,
            (edit.get("suggested_text") or "").strip("\n"),
            edit.get("reason", ""), confidence,
        )
    )


class SaveResponse(BaseModel):
    saved: list[Suggestion]
    files: list[str]


@app.post("/api/sessions/{session_id}/save", response_model=SaveResponse)
def save(session_id: str) -> SaveResponse:
    if not store.get_session(session_id):
        raise HTTPException(status_code=404, detail="session not found")
    t0 = time.perf_counter()
    saved = store.save_session(session_id)
    files = sorted({s.file for s in saved})
    logger.info("POST /save total=%.1fms saved=%d files=%d",
                (time.perf_counter() - t0) * 1000, len(saved), len(files))
    return SaveResponse(saved=saved, files=files)


class DocumentResponse(BaseModel):
    file: str
    content: str
    edited: bool


@app.get("/api/documents/{file:path}", response_model=DocumentResponse)
def get_document(file: str) -> DocumentResponse:
    try:
        content = store.get_document(file)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="document not found") from exc
    return DocumentResponse(file=file, content=content, edited=store.is_edited(file))
