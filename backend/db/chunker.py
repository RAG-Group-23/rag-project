from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_pages(pages, document_id: str, chunk_size: int = 300, chunk_overlap: int = 50) -> list[Document]:
    documents = [
        Document(
            page_content=page.texts or "",
            metadata={
                "document_id": document_id,
                "page": page_number,
                "num_images": len(page.images),
            },
        )
        for page_number, page in enumerate(pages, start=1)
    ]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks = text_splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{document_id}_chunk_{i}"
        chunk.metadata["chunk_index"] = i

    return chunks