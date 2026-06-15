"""Header-based markdown chunker.

Splits each markdown doc into section chunks on heading boundaries while keeping
fenced code blocks intact (a ``#`` inside a ```` ``` ```` block is not a heading).
Sections larger than ``MAX_CHARS`` are sub-split on the next deeper heading level,
and any remaining oversized leaf is hard-split with overlap so no chunk blows past
the embedding/context budget.

Each chunk carries the file path and a 1-based line span so an approved edit can be
written back to the exact location in the original file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE_RE = re.compile(r"^\s*(```+|~~~+)")

# Soft ceiling per chunk (chars). Big sections get sub-split below this.
MAX_CHARS = 4000
# Hard-split overlap (chars) so context is not lost at a forced boundary.
HARD_SPLIT_OVERLAP = 200


@dataclass
class Chunk:
    chunk_id: str
    file: str  # path relative to docs root, e.g. "sessions/index.md"
    doc_title: str  # the document's top-level (h1) title
    heading: str  # this section's heading text
    header_path: list[str]  # breadcrumb of headings, outermost first
    text: str  # full section text including its heading line
    start_line: int  # 1-based, inclusive
    end_line: int  # 1-based, inclusive
    metadata: dict = field(default_factory=dict)

    @property
    def breadcrumb(self) -> str:
        return " > ".join(self.header_path) if self.header_path else self.doc_title


@dataclass
class _Section:
    level: int
    heading: str
    header_path: list[str]
    lines: list[str]
    start_line: int  # 1-based


def _split_into_sections(lines: list[str]) -> list[_Section]:
    """Walk lines tracking code-fence state; cut a new section at every heading."""
    sections: list[_Section] = []
    path_stack: list[tuple[int, str]] = []  # (level, heading)
    in_fence = False
    fence_marker = ""

    current = _Section(level=0, heading="", header_path=[], lines=[], start_line=1)

    for idx, line in enumerate(lines):
        fence_match = FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)[:3]
            if not in_fence:
                in_fence, fence_marker = True, marker
            elif line.strip().startswith(fence_marker):
                in_fence = False
            current.lines.append(line)
            continue

        heading_match = None if in_fence else HEADING_RE.match(line)
        if heading_match:
            if current.lines:
                sections.append(current)
            level = len(heading_match.group(1))
            heading = heading_match.group(2).strip()
            while path_stack and path_stack[-1][0] >= level:
                path_stack.pop()
            path_stack.append((level, heading))
            current = _Section(
                level=level,
                heading=heading,
                header_path=[h for _, h in path_stack],
                lines=[line],
                start_line=idx + 1,
            )
        else:
            current.lines.append(line)

    if current.lines:
        sections.append(current)
    return sections


def _hard_split(text: str) -> list[str]:
    if len(text) <= MAX_CHARS:
        return [text]
    parts: list[str] = []
    step = MAX_CHARS - HARD_SPLIT_OVERLAP
    start = 0
    while start < len(text):
        parts.append(text[start : start + MAX_CHARS])
        start += step
    return parts


def chunk_markdown(text: str, file: str) -> list[Chunk]:
    lines = text.splitlines()
    if not lines:
        return []

    sections = _split_into_sections(lines)
    doc_title = next(
        (s.heading for s in sections if s.level == 1 and s.heading),
        Path(file).stem.replace("_", " ").title(),
    )

    chunks: list[Chunk] = []
    for s in sections:
        body = "\n".join(s.lines).strip("\n")
        if not body.strip():
            continue
        end_line = s.start_line + len(s.lines) - 1
        header_path = s.header_path or [doc_title]

        pieces = _hard_split(body)
        for i, piece in enumerate(pieces):
            suffix = f".{i}" if len(pieces) > 1 else ""
            chunks.append(
                Chunk(
                    chunk_id=f"{file}::{s.start_line}{suffix}",
                    file=file,
                    doc_title=doc_title,
                    heading=s.heading or doc_title,
                    header_path=header_path,
                    text=piece,
                    start_line=s.start_line,
                    end_line=end_line,
                )
            )
    return chunks


def load_and_chunk(docs_root: Path) -> list[Chunk]:
    """Chunk every ``*.md`` under ``docs_root`` (skips llms*.txt bundles)."""
    chunks: list[Chunk] = []
    for md_path in sorted(docs_root.rglob("*.md")):
        rel = md_path.relative_to(docs_root).as_posix()
        text = md_path.read_text(encoding="utf-8")
        chunks.extend(chunk_markdown(text, rel))
    return chunks
