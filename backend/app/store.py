"""In-memory state: review sessions and the edited-document overlay.

TRADEOFF (documented in README): everything here lives in process memory and is
lost on restart. Production would persist sessions/suggestions in Postgres and
write approved edits to the docs repo via a branch + pull request rather than an
in-memory overlay.

Saving applies each approved suggestion by replacing the chunk's exact original
text with the edited text inside a working copy of the file. Exact-text replace is
order-independent, so multiple edits to one file don't fight over shifting line
numbers. The original files on disk are never mutated, so the demo corpus stays
reproducible across queries.
"""

from __future__ import annotations

import threading

from . import config
from .models import SessionState, Suggestion, SuggestionStatus


class Store:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sessions: dict[str, SessionState] = {}
        self._overlay: dict[str, str] = {}  # file -> edited contents

    # ---- sessions --------------------------------------------------------
    def create_session(self, session_id: str, query: str, suggestions: list[Suggestion]) -> SessionState:
        state = SessionState(session_id=session_id, query=query, suggestions=suggestions)
        with self._lock:
            self._sessions[session_id] = state
        return state

    def get_session(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

    def list_sessions(self) -> list[SessionState]:
        return list(self._sessions.values())

    def _find(self, session_id: str, suggestion_id: str) -> Suggestion | None:
        session = self._sessions.get(session_id)
        if not session:
            return None
        return next((s for s in session.suggestions if s.id == suggestion_id), None)

    def set_status(self, session_id: str, suggestion_id: str, status: SuggestionStatus) -> Suggestion | None:
        with self._lock:
            sug = self._find(session_id, suggestion_id)
            if sug:
                sug.status = status
            return sug

    def edit_suggestion(self, session_id: str, suggestion_id: str, suggested_text: str) -> Suggestion | None:
        with self._lock:
            sug = self._find(session_id, suggestion_id)
            if sug:
                sug.suggested_text = suggested_text
                sug.status = SuggestionStatus.approved
            return sug

    def replace_suggestion(
        self, session_id: str, suggestion_id: str,
        suggested_text: str, reason: str, confidence,
    ) -> Suggestion | None:
        """Overwrite a suggestion's edit with a freshly regenerated one; reset to pending."""
        with self._lock:
            sug = self._find(session_id, suggestion_id)
            if sug:
                sug.suggested_text = suggested_text
                sug.reason = reason
                sug.confidence = confidence
                sug.status = SuggestionStatus.pending
            return sug

    def session_query(self, session_id: str) -> str | None:
        s = self._sessions.get(session_id)
        return s.query if s else None

    # ---- saving ----------------------------------------------------------
    def save_session(self, session_id: str) -> list[Suggestion]:
        """Apply approved suggestions to the overlay and write edited file copies."""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return []
            saved: list[Suggestion] = []
            for sug in session.suggestions:
                if sug.status != SuggestionStatus.approved:
                    continue
                content = self._current_content(sug.file)
                if sug.original_text in content:
                    content = content.replace(sug.original_text, sug.suggested_text, 1)
                    self._overlay[sug.file] = content
                    self._write_result_file(sug.file, content)
                    sug.status = SuggestionStatus.saved
                    saved.append(sug)
                else:
                    # Original drifted (e.g. an earlier overlapping save). Skip
                    # rather than corrupt the file; surfaced to the caller by
                    # omission from the saved list.
                    continue
            return saved

    def _write_result_file(self, file: str, content: str) -> None:
        out = config.RESULT_DIR / file
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content, encoding="utf-8")

    def _current_content(self, file: str) -> str:
        if file in self._overlay:
            return self._overlay[file]
        return (config.DOCS_ROOT / file).read_text(encoding="utf-8")

    def get_document(self, file: str) -> str:
        return self._current_content(file)

    def is_edited(self, file: str) -> bool:
        return file in self._overlay


store = Store()
