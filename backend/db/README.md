# Document Chunker

This module chunks extracted research-paper PDFs into LangChain `Document` objects for vector indexing.

## Approach

The chunker is research-paper aware:

1. It preserves metadata such as `document_id`, `page`, `section`, `chunk_id`, and `num_images`.
2. It detects likely section headings such as `Abstract`, `Introduction`, `3.2 Attention`, etc.
3. It splits text by sections and paragraphs first.
4. It merges nearby paragraphs into chunks.
5. It uses LangChain's `RecursiveCharacterTextSplitter` only as a fallback when a section/paragraph block is still too large.

## Why not only fixed-size splitting?

Pure fixed-size splitting can cut through paragraphs, equations, or explanations. For research papers, section and paragraph boundaries are usually meaningful, so we preserve them when possible.

## Why still use RecursiveCharacterTextSplitter?

LangChain recommends `RecursiveCharacterTextSplitter` as a strong generic fallback because it tries to keep paragraphs, then sentences, then words together while enforcing a maximum chunk size. This makes it useful as a final size-control step, not as the only chunking strategy.

Sources:
- LangChain docs: RecursiveCharacterTextSplitter is recommended for generic text and tries to preserve paragraphs/sentences/words.
- LangChain text splitter docs: start with RecursiveCharacterTextSplitter for most use cases.
- Pinecone RAG chunking guide: semantic/meaningful chunking groups related sentences and detects topic shifts.