from __future__ import annotations

import re
from dataclasses import dataclass, field
from io import IOBase
from pathlib import Path

import pymupdf
import pymupdf4llm


# ---------------------------------------------------------------------------
# Public data structures
# ---------------------------------------------------------------------------

@dataclass
class Section:
    """One logical section extracted from a Markdown-converted PDF."""
    title: str | None   # None for content before the first heading
    level: int          # 1 for #, 2 for ##, 0 for preamble content
    text: str           # body text, heading line excluded
    page_start: int     # 1-indexed
    page_end: int       # 1-indexed; inclusive
    # Per-character page mapping: contiguous list of (start, end_exclusive, page)
    # covering [0, len(text)) when text is non-empty. Same-page consecutive runs
    # are coalesced. Used by the chunker to derive per-chunk page numbers when
    # a section spans multiple pages.
    page_spans: list[tuple[int, int, int]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal Markdown parser
# ---------------------------------------------------------------------------

# ATX heading: optional leading whitespace, 1–6 #, one space, title text.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")


def _trim_blank_lines(lines_with_pages: list[tuple[str, int]]) -> list[tuple[str, int]]:
    """Drop leading/trailing entries whose line is empty or whitespace-only."""
    start = 0
    end = len(lines_with_pages)
    while start < end and not lines_with_pages[start][0].strip():
        start += 1
    while end > start and not lines_with_pages[end - 1][0].strip():
        end -= 1
    return lines_with_pages[start:end]


def _build_text_and_spans(
    lines_with_pages: list[tuple[str, int]],
) -> tuple[str, list[tuple[int, int, int]]]:
    """Join ``(line, page)`` pairs on ``"\\n"`` and return ``(text, page_spans)``.

    ``page_spans`` contiguously covers ``[0, len(text))`` with consecutive
    same-page runs coalesced; the inter-line ``"\\n"`` separator is absorbed
    into the preceding span so spans abut exactly.
    """
    if not lines_with_pages:
        return "", []

    line_spans: list[tuple[int, int, int]] = []
    parts: list[str] = []
    offset = 0
    for i, (line, page) in enumerate(lines_with_pages):
        if i > 0:
            parts.append("\n")
            offset += 1
        start = offset
        parts.append(line)
        offset += len(line)
        line_spans.append((start, offset, page))

    spans: list[tuple[int, int, int]] = []
    for s, e, p in line_spans:
        if spans and spans[-1][2] == p:
            spans[-1] = (spans[-1][0], e, p)
        else:
            if spans:
                prev = spans[-1]
                spans[-1] = (prev[0], s, prev[2])
            spans.append((s, e, p))

    return "".join(parts), spans


def _parse_page_chunks(page_chunks: list[dict]) -> list[Section]:
    """Parse pymupdf4llm ``page_chunks=True`` output into :class:`Section`s."""
    sections: list[Section] = []
    current_title: str | None = None
    current_level: int = 0
    current_lines: list[tuple[str, int]] = []
    section_page_start: int = 1
    last_page: int = 1

    def _flush(end_page: int) -> None:
        trimmed = _trim_blank_lines(current_lines)
        text, spans = _build_text_and_spans(trimmed)
        if text or current_title is not None:
            sections.append(Section(
                title=current_title,
                level=current_level,
                text=text,
                page_start=section_page_start,
                page_end=end_page,
                page_spans=spans,
            ))

    for idx, entry in enumerate(page_chunks):
        meta = entry.get("metadata", {}) or {}
        page_num = meta.get("page") or (idx + 1)
        last_page = page_num
        if not sections and current_title is None and not current_lines:
            section_page_start = page_num

        for line in (entry.get("text", "") or "").splitlines():
            m = _HEADING_RE.match(line)
            if m:
                _flush(page_num)
                current_title = m.group(2).strip()
                current_level = len(m.group(1))
                current_lines = []
                section_page_start = page_num
            else:
                current_lines.append((line, page_num))

    _flush(last_page)
    return sections


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_sections(source: str | Path | bytes | bytearray | IOBase) -> list[Section]:
    """
    Extract logical sections from a PDF by converting it to Markdown first.

    Uses ``pymupdf4llm.to_markdown(..., page_chunks=True)`` so page numbers
    come from PyMuPDF directly (per-page metadata) instead of being inferred
    from in-text separators.

    Parameters
    ----------
    source:
        Either a filesystem path to a PDF, or in-memory PDF bytes / a
        binary file-like object (e.g. ``BytesIO``).

    Returns
    -------
    list[Section]
        Sections in document order.  The first entry may have ``title=None``
        for any content appearing before the first heading.  Each section
        carries a ``page_spans`` mapping from character offsets within
        ``text`` to source page numbers.
    """
    if isinstance(source, (str, Path)):
        doc = pymupdf.open(str(source))
    elif isinstance(source, (bytes, bytearray)):
        doc = pymupdf.open(stream=bytes(source), filetype="pdf")
    elif isinstance(source, IOBase):
        source.seek(0)
        doc = pymupdf.open(stream=source.read(), filetype="pdf")
    else:
        raise TypeError(
            f"extract_sections: unsupported source type {type(source).__name__}"
        )

    try:
        page_chunks = pymupdf4llm.to_markdown(doc, page_chunks=True)
    finally:
        doc.close()

    return _parse_page_chunks(page_chunks)
