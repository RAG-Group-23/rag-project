import re
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


SECTION_HEADING_PATTERN = re.compile(
    r"^\s*(?:"
    r"(?:\d+(?:\.\d+)*\.?\s+[A-Z][^\n]{2,120})"
    r"|(?:Abstract|Introduction|Background|Related Work|Method|Methods|Methodology|"
    r"Experiments|Results|Discussion|Conclusion|References|Appendix)\s*$"
    r")",
    re.IGNORECASE,
)


def _clean_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"-\n(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_section_heading(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    return bool(SECTION_HEADING_PATTERN.match(line))


def _split_page_into_sections(text: str) -> list[tuple[str | None, str]]:
    lines = text.splitlines()
    sections: list[tuple[str | None, list[str]]] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        if _is_section_heading(stripped):
            if current_lines:
                sections.append((current_heading, current_lines))
            current_heading = stripped
            current_lines = [stripped]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, current_lines))

    return [
        (heading, "\n".join(section_lines).strip())
        for heading, section_lines in sections
        if "\n".join(section_lines).strip()
    ]


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _merge_paragraphs(
    paragraphs: Iterable[str],
    max_chars: int,
    overlap_chars: int,
) -> list[str]:
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if not current:
            current = paragraph
            continue

        candidate = current + "\n\n" + paragraph

        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)

            if overlap_chars > 0:
                overlap = current[-overlap_chars:]
                current = overlap + "\n\n" + paragraph
            else:
                current = paragraph

    if current:
        chunks.append(current)

    return chunks


def chunk_pages(
    pages,
    document_id: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> list[Document]:
    """
    Chunk extracted research-paper pages into LangChain Documents.

    Strategy:
    1. Preserve document/page metadata.
    2. Detect likely research-paper section headings.
    3. Split pages into section-aware blocks.
    4. Split sections into paragraphs.
    5. Merge paragraphs into chunks.
    6. Use RecursiveCharacterTextSplitter only as fallback for oversized chunks.
    """
    initial_chunks: list[Document] = []

    for page_number, page in enumerate(pages, start=1):
        page_text = _clean_text(getattr(page, "texts", "") or "")
        if not page_text:
            continue

        sections = _split_page_into_sections(page_text)

        for section_index, (section_title, section_text) in enumerate(sections):
            paragraphs = _split_paragraphs(section_text)
            merged_chunks = _merge_paragraphs(
                paragraphs=paragraphs,
                max_chars=chunk_size,
                overlap_chars=chunk_overlap,
            )

            for local_chunk_index, chunk_text in enumerate(merged_chunks):
                initial_chunks.append(
                    Document(
                        page_content=chunk_text,
                        metadata={
                            "document_id": document_id,
                            "page_index": page_number,
                            "section": section_title,
                            "section_index": section_index,
                            "local_chunk_index": local_chunk_index,
                            "num_images": len(getattr(page, "images", [])),
                        },
                    )
                )

    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    final_chunks = fallback_splitter.split_documents(initial_chunks)

    for chunk_index, chunk in enumerate(final_chunks):
        chunk.metadata["chunk_id"] = f"{document_id}_chunk_{chunk_index}"
        chunk.metadata["chunk_index"] = chunk_index

    return final_chunks
