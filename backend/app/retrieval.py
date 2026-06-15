"""Embedding index + cosine retrieval over the doc chunks.

On startup we chunk every doc, embed each chunk with OpenAI, and keep the matrix
in memory. Embeddings are cached to disk keyed by a hash of (model + chunk text),
so a restart with unchanged docs costs no API calls. A query is embedded once and
scored against the matrix by cosine similarity.

Why cosine over a vector DB: the whole corpus is ~110k tokens / a few hundred
chunks. A NumPy dot-product is exact, instant, and dependency-light. See README
for when this should become a real vector store.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from openai import OpenAI

from . import config
from .chunking import Chunk, load_and_chunk

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9_]+")

# Minimal English stoplist. Change requests are full sentences ("we don't support
# X anymore, other agents should only..."); without this, BM25 ranks on filler.
_STOPWORDS = {
    "the", "and", "for", "are", "was", "were", "you", "your", "our", "any", "all",
    "can", "could", "should", "would", "will", "shall", "may", "might", "must",
    "this", "that", "these", "those", "with", "from", "into", "via", "out", "off",
    "not", "now", "anymore", "only", "other", "others", "than", "then", "them",
    "they", "their", "have", "has", "had", "does", "did", "doing", "done", "but",
    "what", "when", "where", "which", "who", "why", "how", "use", "used", "using",
    "want", "wants", "need", "needs", "support", "supports", "change", "changed",
    "update", "updates", "updated", "longer", "anymore", "dont", "don", "isnt",
    "we", "i", "it", "is", "be", "to", "of", "in", "on", "or", "an", "as", "at",
    "by", "no", "so", "if", "do",
}


def _tokenize(text: str, drop_stopwords: bool = False) -> list[str]:
    # Lowercase word/identifier tokens. Underscores kept, so ``as_tool`` survives
    # as one token (helps API-symbol recall); length-1 tokens dropped as noise.
    toks = [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1]
    if drop_stopwords:
        toks = [t for t in toks if t not in _STOPWORDS]
    return toks


class BM25:
    """Okapi BM25 lexical retriever over the chunk corpus.

    Classic keyword retrieval: rare query terms that appear often in a chunk score
    high. Complements dense (semantic) retrieval on exact wording the embedding may
    not surface. Pure-Python — corpus is small, so no index server needed.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunk_ids: list[str] = []
        self.doc_tokens: list[list[str]] = []
        self.doc_len: list[int] = []
        self.avgdl: float = 0.0
        self.idf: dict[str, float] = {}
        self.tf: list[Counter] = []

    def fit(self, chunks: list[Chunk]) -> None:
        self.chunk_ids = [c.chunk_id for c in chunks]
        self.doc_tokens = [_tokenize(c.text) for c in chunks]
        self.tf = [Counter(toks) for toks in self.doc_tokens]
        self.doc_len = [len(toks) for toks in self.doc_tokens]
        n = len(chunks)
        self.avgdl = (sum(self.doc_len) / n) if n else 0.0

        df: dict[str, int] = defaultdict(int)
        for toks in self.doc_tokens:
            for term in set(toks):
                df[term] += 1
        # BM25+ idf (always positive) to avoid negative scores on common terms.
        self.idf = {t: math.log(1 + (n - dfi + 0.5) / (dfi + 0.5)) for t, dfi in df.items()}

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        q_terms = [t for t in _tokenize(query, drop_stopwords=True) if t in self.idf]
        if not q_terms or not self.chunk_ids:
            return []
        scores: list[tuple[str, float]] = []
        for i, cid in enumerate(self.chunk_ids):
            tf_i, dl = self.tf[i], self.doc_len[i]
            denom_norm = self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            s = 0.0
            for term in q_terms:
                f = tf_i.get(term, 0)
                if f:
                    s += self.idf[term] * (f * (self.k1 + 1)) / (f + denom_norm)
            if s > 0:
                scores.append((cid, s))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Merge several ranked id lists into one. RRF score = Σ 1/(k + rank).

    Rank-based, so dense cosine and BM25 (incomparable raw scores) fuse fairly: an
    id ranked high in *either* list rises. k=60 is the standard damping constant.
    """
    fused: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            fused[cid] += 1.0 / (k + rank + 1)
    return sorted(fused.items(), key=lambda x: -x[1])

_EMBED_BATCH = 128

# Code-symbol shapes worth grepping for verbatim: `backtick spans`, dotted paths
# (agents.tool.WebSearchTool), snake_case (as_tool), and CamelCase (ApplyPatchTool).
_SYMBOL_RE = re.compile(
    r"`([^`]+)`"
    r"|\b([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)+)\b"
    r"|\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b"
    r"|\b([A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+)\b"
)
# snake_case words that are ordinary English, not API symbols.
_SYMBOL_STOPWORDS = {"e_g", "i_e", "etc"}


_QUOTED_RE = re.compile(r"['\"]([A-Za-z_][\w .-]{1,40})['\"]")


def extract_symbols(query: str) -> list[str]:
    """Pull likely code symbols out of a natural-language change request.

    These are the exact tokens embeddings most often miss (a rename query says
    ``as_tool``; the doc prose says "expose an agent as a callable tool"). We grep
    for them verbatim so the relevant code samples are never dropped.

    Also captures single/double-quoted terms (``rename 'handoff' to 'transfer'``)
    so plain-word terminology changes — which the code-symbol regex misses — are
    still found.
    """
    symbols: list[str] = []
    seen: set[str] = set()

    def add(token: str) -> None:
        token = token.split("(")[0].strip()
        if len(token) < 3 or token.lower() in _SYMBOL_STOPWORDS:
            return
        if token not in seen:
            seen.add(token)
            symbols.append(token)

    for match in _SYMBOL_RE.finditer(query):
        add(next((g for g in match.groups() if g), "").strip())
    for match in _QUOTED_RE.finditer(query):
        add(match.group(1).strip())
    return symbols


def _cache_key(text: str) -> str:
    h = hashlib.sha256(f"{config.EMBED_MODEL}\x00{text}".encode()).hexdigest()
    return h


class DocIndex:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self.matrix: np.ndarray | None = None  # (n_chunks, dim), L2-normalized
        self.by_id: dict[str, Chunk] = {}
        self.bm25 = BM25()
        self._query_cache: dict[str, np.ndarray] = {}
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            if not config.OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY is not set")
            self._client = OpenAI(api_key=config.OPENAI_API_KEY)
        return self._client

    # ---- build -----------------------------------------------------------
    def build(self) -> None:
        self.chunks = load_and_chunk(config.DOCS_ROOT)
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.bm25.fit(self.chunks)
        if not self.chunks:
            logger.warning("No chunks found under %s", config.DOCS_ROOT)
            self.matrix = np.zeros((0, 1), dtype=np.float32)
            return

        vectors = self._embed_chunks(self.chunks)
        self.matrix = self._normalize(np.asarray(vectors, dtype=np.float32))
        logger.info("Indexed %d chunks (dim=%d)", len(self.chunks), self.matrix.shape[1])

    def _embed_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        cache = _DiskCache(config.CACHE_DIR / "embeddings")
        results: list[list[float] | None] = [None] * len(chunks)
        missing: list[int] = []

        for i, c in enumerate(chunks):
            cached = cache.get(_cache_key(c.text))
            if cached is not None:
                results[i] = cached
            else:
                missing.append(i)

        for start in range(0, len(missing), _EMBED_BATCH):
            batch_idx = missing[start : start + _EMBED_BATCH]
            inputs = [self._embed_input(chunks[i]) for i in batch_idx]
            resp = self.client.embeddings.create(model=config.EMBED_MODEL, input=inputs)
            for j, item in zip(batch_idx, resp.data):
                results[j] = item.embedding
                cache.set(_cache_key(chunks[j].text), item.embedding)
            logger.info("Embedded %d/%d new chunks", start + len(batch_idx), len(missing))

        cache.flush()
        return [r for r in results if r is not None]

    @staticmethod
    def _embed_input(chunk: Chunk) -> str:
        # Prepend the breadcrumb so a short section still carries its context.
        return f"{chunk.file} | {chunk.breadcrumb}\n\n{chunk.text}"

    @staticmethod
    def _normalize(m: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return m / norms

    # ---- search ----------------------------------------------------------
    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a query, memoized per-process. The same query is retrieved by
        several variants (eval) and on repeat user requests; embedding it once
        saves API calls and avoids hammering the endpoint."""
        cached = self._query_cache.get(query)
        if cached is not None:
            return cached
        resp = self.client.embeddings.create(model=config.EMBED_MODEL, input=[query])
        q = np.asarray(resp.data[0].embedding, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        self._query_cache[query] = q
        return q

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        if self.matrix is None or self.matrix.shape[0] == 0:
            return []
        q = self._embed_query(query)
        scores = self.matrix @ q
        top = np.argsort(-scores)[: min(k, len(self.chunks))]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def symbol_search(self, query: str, limit: int = 20) -> list[Chunk]:
        """Chunks containing a code symbol named in the query (verbatim match).

        Pure recall booster: only adds candidates, never removes embedding hits.
        Case-sensitive so ``as_tool`` does not match unrelated prose. Ranked by how
        many distinct query symbols a chunk mentions.
        """
        symbols = extract_symbols(query)
        if not symbols:
            return []
        hits: list[tuple[int, Chunk]] = []
        for chunk in self.chunks:
            count = sum(1 for s in symbols if s in chunk.text)
            if count:
                hits.append((count, chunk))
        hits.sort(key=lambda h: -h[0])
        return [c for _, c in hits[:limit]]

    def term_matches(self, query: str) -> list[Chunk]:
        """EVERY chunk containing a term from the query — uncapped.

        For completeness on cross-cutting changes ("rename X everywhere"): a
        top-K cap structurally cannot surface all affected places, so we grep
        exhaustively for the exact symbol/term. Returned in document order.
        """
        terms = extract_symbols(query)
        if not terms:
            return []
        return [c for c in self.chunks if any(t in c.text for t in terms)]


class _DiskCache:
    """Append-friendly JSON cache. Tiny corpus -> a single file is fine."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, list[float]] = {}
        self._dirty = False
        if path.exists():
            try:
                self._data = json.loads(path.read_text())
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def get(self, key: str) -> list[float] | None:
        return self._data.get(key)

    def set(self, key: str, value: list[float]) -> None:
        self._data[key] = value
        self._dirty = True

    def flush(self) -> None:
        if self._dirty:
            self.path.write_text(json.dumps(self._data))
            self._dirty = False


index = DocIndex()
