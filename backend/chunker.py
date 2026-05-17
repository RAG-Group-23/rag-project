"""
Section-aware chunker for research-paper PDFs.

Consumes the output of ``extractor.extract_sections`` — a list of
:class:`~extractor.Section` objects — and produces LangChain ``Document``
objects ready for embedding and retrieval.

All size limits (``chunk_size``, ``chunk_overlap``, ``min_section_tokens``)
are expressed in **tokens**, counted via ``tiktoken``.  This makes the limits
directly meaningful for any downstream model or embedding API regardless of
the character-to-token ratio of the text.

Chunking strategy per section
------------------------------
1. **Fits** (tokens <= chunk_size): emit as a single Document.
2. **Oversized**: split into paragraphs → greedily merge up to *chunk_size*
   tokens with paragraph-level overlap → ``RecursiveCharacterTextSplitter``
   (tiktoken-based) as a last-resort fallback for any individually oversized
   paragraph (e.g. a long table or unbroken figure caption).

Additionally, **undersized sections are merged** with their successor before
chunking when both are below *min_section_tokens*.

Output metadata
---------------
    chunk_id            str   "{document_id}_chunk_{n}"
    chunk_index         int   global position in the final list
    document_id         str
    section             str | None
    section_level       int   1 = #, 2 = ##, …
    section_index       int   position of the section (after any merging)
    local_chunk_index   int   position of this chunk within its section
    page_start          int   first page the chunk's text occupies
    page_end            int   last page the chunk's text occupies (inclusive)
    page                int   single best page for citation: equal to
                              page_start when the chunk lives on one page,
                              otherwise the page contributing the most
                              characters (dominant page)
    num_images          int   always 0 (images not carried through Markdown path)
"""

from __future__ import annotations

import re
from functools import lru_cache

import tiktoken
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _get_encoding(model: str) -> tiktoken.Encoding:
    """Return (and cache) the tiktoken encoding for *model*."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        # Fall back to the broadly-compatible cl100k_base encoding.
        return tiktoken.get_encoding("cl100k_base")


def _count_tokens(text: str, encoding: tiktoken.Encoding) -> int:
    return len(encoding.encode(text))


def _pages_for_range(
    page_spans: list[tuple[int, int, int]], start: int, end: int
) -> tuple[int | None, int | None]:
    """Return (min_page, max_page) of pages overlapping ``[start, end)``."""
    pages: set[int] = set()
    for s, e, p in page_spans:
        if e <= start or s >= end:
            continue
        pages.add(p)
    if not pages:
        return None, None
    return min(pages), max(pages)


def _dominant_page(
    page_spans: list[tuple[int, int, int]], start: int, end: int
) -> int | None:
    """Return the page contributing the most characters to ``[start, end)``.

    Ties are broken by the lower page number (earlier in the document).
    """
    counts: dict[int, int] = {}
    for s, e, p in page_spans:
        overlap = max(0, min(e, end) - max(s, start))
        if overlap:
            counts[p] = counts.get(p, 0) + overlap
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]


# ---------------------------------------------------------------------------
# Paragraph splitting and token-aware merging
# ---------------------------------------------------------------------------

def _split_paragraphs(text: str) -> list[str]:
    """Split *text* on blank lines, returning non-empty paragraph strings."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _merge_paragraphs(
    paragraphs: list[str],
    max_tokens: int,
    overlap_tokens: int,
    encoding: tiktoken.Encoding,
) -> list[str]:
    """
    Greedily merge *paragraphs* into chunks of at most *max_tokens*.

    Token counts are computed once per paragraph and cached in a local list to
    avoid re-encoding on every iteration.

    Overlap is seeded from the last *whole paragraph* that was flushed,
    provided its token count fits within *overlap_tokens*.  This keeps overlap
    coherent at paragraph boundaries rather than slicing mid-token.
    """
    # Pre-compute token counts; +2 tokens budgeted for the "\n\n" separator.
    sep_tokens = _count_tokens("\n\n", encoding)
    para_tokens = [_count_tokens(p, encoding) for p in paragraphs]

    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens: int = 0

    def _join(parts: list[str]) -> str:
        return "\n\n".join(parts)

    for para, tokens in zip(paragraphs, para_tokens):
        added = tokens + (sep_tokens if current_parts else 0)

        if current_parts and current_tokens + added > max_tokens:
            chunks.append(_join(current_parts))
            last_para = current_parts[-1]
            last_tokens = para_tokens[paragraphs.index(last_para)]

            if overlap_tokens > 0 and last_para != para and last_tokens <= overlap_tokens:
                current_parts = [last_para, para]
                current_tokens = last_tokens + sep_tokens + tokens
            else:
                current_parts = [para]
                current_tokens = tokens
        else:
            current_parts.append(para)
            current_tokens += added

    if current_parts:
        chunks.append(_join(current_parts))

    return chunks


# ---------------------------------------------------------------------------
# Section merging
# ---------------------------------------------------------------------------

def _merge_small_sections(sections, min_tokens: int, encoding: tiktoken.Encoding):
    """
    Merge consecutive sections that are individually below *min_tokens*.

    A short section is absorbed into its successor until the combined token
    count crosses *min_tokens* or there are no more sections to absorb.  The
    merged section keeps the title and level of the first section in the group.
    """
    if not sections:
        return sections

    from text_extractor import Section  # local import to avoid circular dependency

    merged: list[Section] = []
    buffer: Section | None = None
    buffer_tokens: int = 0

    for section in sections:
        section_tokens = _count_tokens(section.text, encoding)

        if buffer is None:
            buffer = section
            buffer_tokens = section_tokens
            continue

        if buffer_tokens < min_tokens and buffer_tokens + section_tokens <= min_tokens * 4:
            from text_extractor import Section as _Section
            sep = "\n\n"
            new_text = buffer.text + sep + section.text
            shift = len(buffer.text) + len(sep)
            new_spans = list(buffer.page_spans) + [
                (s + shift, e + shift, p) for (s, e, p) in section.page_spans
            ]
            buffer = _Section(
                title=buffer.title,
                level=buffer.level,
                text=new_text,
                page_start=buffer.page_start,
                page_end=section.page_end,
                page_spans=new_spans,
            )
            buffer_tokens = _count_tokens(buffer.text, encoding)
        else:
            merged.append(buffer)
            buffer = section
            buffer_tokens = section_tokens

    if buffer is not None:
        merged.append(buffer)

    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_sections(
    sections,
    document_id: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    min_section_tokens: int = 64,
    tiktoken_model: str = "text-embedding-3-small",
) -> list[Document]:
    """
    Chunk a list of :class:`~extractor.Section` objects into LangChain
    ``Document`` objects suitable for embedding and retrieval.

    Parameters
    ----------
    sections:
        Output of ``extractor.extract_sections``.
    document_id:
        Stable identifier for the source document (used in ``chunk_id``).
    chunk_size:
        Maximum tokens per chunk.
    chunk_overlap:
        Maximum tokens of overlap between consecutive chunks from the same
        section (implemented at paragraph granularity).
    min_section_tokens:
        Sections shorter than this are merged with their successor before
        chunking.  Set to 0 to disable merging.
    tiktoken_model:
        Model name passed to ``tiktoken.encoding_for_model``.  Defaults to
        ``"text-embedding-3-small"`` (cl100k_base).  Use the name of whatever
        embedding model you are targeting so token counts align exactly.

    Returns
    -------
    list[Document]
        Chunks in document order with fully populated metadata.
    """
    encoding = _get_encoding(tiktoken_model)

    if min_section_tokens > 0:
        sections = _merge_small_sections(
            sections, min_section_tokens, encoding)

    initial_chunks: list[Document] = []

    for section_index, section in enumerate(sections):
        if not section.text:
            continue

        section_tokens = _count_tokens(section.text, encoding)

        base_meta = {
            "document_id": document_id,
            "section": section.title,
            "section_level": section.level,
            "section_index": section_index,
            "page_start": section.page_start,
            "page_end": section.page_end,
            "num_images": 0,
        }

        if section_tokens <= chunk_size:
            initial_chunks.append(Document(
                page_content=section.text,
                metadata={**base_meta, "local_chunk_index": 0},
            ))
            continue

        paragraphs = _split_paragraphs(section.text)
        merged = _merge_paragraphs(
            paragraphs, chunk_size, chunk_overlap, encoding)

        for local_index, chunk_text in enumerate(merged):
            initial_chunks.append(Document(
                page_content=chunk_text,
                metadata={**base_meta, "local_chunk_index": local_index},
            ))

    # Last-resort fallback for any chunk still over chunk_size (e.g. a single
    # paragraph that is an enormous table or unbroken figure caption).
    # from_tiktoken_encoder uses the same tokeniser so the limit is consistent.
    # strip_whitespace=False keeps chunks as exact substrings of the section
    # text, which the post-pass below relies on for accurate page lookup.
    fallback = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=tiktoken_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        strip_whitespace=False,
    )
    final_chunks = fallback.split_documents(initial_chunks)

    # Refine per-chunk page metadata using each section's page_spans. Chunks
    # are emitted in document order; a per-section running search pointer
    # disambiguates repeated substrings (e.g. when paragraph-level overlap
    # makes consecutive chunks share text). Every chunk gets a single ``page``
    # value: the dominant (most-overlapped) page when it spans a boundary, and
    # the section's start page as a last-resort fallback if substring lookup
    # fails.
    cur_sec_idx: int | None = None
    search_pos = 0
    for chunk in final_chunks:
        sec_idx = chunk.metadata.get("section_index")
        if sec_idx != cur_sec_idx:
            cur_sec_idx = sec_idx
            search_pos = 0
        if sec_idx is None or sec_idx >= len(sections):
            continue
        section = sections[sec_idx]
        if not section.page_spans:
            chunk.metadata.setdefault("page", section.page_start)
            continue
        text = chunk.page_content
        pos = section.text.find(text, search_pos)
        if pos < 0:
            pos = section.text.find(text)
        if pos < 0:
            chunk.metadata.setdefault("page", section.page_start)
            continue
        ps, pe = _pages_for_range(
            section.page_spans, pos, pos + len(text))
        if ps is not None:
            chunk.metadata["page_start"] = ps
            chunk.metadata["page_end"] = pe
            dom = _dominant_page(
                section.page_spans, pos, pos + len(text))
            chunk.metadata["page"] = dom if dom is not None else ps
        else:
            chunk.metadata.setdefault("page", section.page_start)
        search_pos = pos + 1

    for i, chunk in enumerate(final_chunks):
        chunk.metadata["chunk_id"] = f"{document_id}_chunk_{i}"
        chunk.metadata["chunk_index"] = i

    return final_chunks