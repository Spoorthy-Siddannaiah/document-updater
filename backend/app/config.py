"""Runtime configuration, read from environment."""

from __future__ import annotations

import os
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
DOCS_ROOT = Path(os.getenv("DOCS_ROOT", BACKEND_ROOT / "data" / "agents-sdk-docs"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", BACKEND_ROOT / ".cache"))
RESULT_DIR = Path(os.getenv("RESULT_DIR", BACKEND_ROOT / "result"))
LOG_DIR = Path(os.getenv("LOG_DIR", BACKEND_ROOT / "log"))
LOG_FILE = Path(os.getenv("LOG_FILE", LOG_DIR / "rag.log"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o")

# How many candidate chunks retrieval feeds to the suggestion model.
TOP_K = int(os.getenv("TOP_K", "12"))
# Depth pulled from each retriever before RRF fusion.
DENSE_DEPTH = int(os.getenv("DENSE_DEPTH", "20"))
BM25_DEPTH = int(os.getenv("BM25_DEPTH", "20"))
# Symbol grep force-include. Eval showed it adds ~33% candidates for zero recall/MRR
# gain once dense+BM25 RRF is in place (it's subsumed). Default off; flip on for
# corpora where exact code-symbol coverage matters more than token cost.
USE_SYMBOL_GREP = os.getenv("USE_SYMBOL_GREP", "0") == "1"
# Max candidate sections per LLM call. Cross-cutting changes produce many
# candidates; we batch so every affected section is reviewed, not just the first.
SUGGEST_BATCH = int(os.getenv("SUGGEST_BATCH", "4"))
# Batches sent to the LLM concurrently. Cuts wall-clock latency on cross-cutting
# changes from sum-of-batches to ~slowest-batch.
MAX_PARALLEL_BATCHES = int(os.getenv("MAX_PARALLEL_BATCHES", "6"))

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
