import os
from io import BytesIO
from typing import Any, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vectorstore import VectorDBInterface, PGVectorDBInstance, ChromaDBInstance
from text_extractor import extract_text_and_images_from_paper


# ----------------------------------------
# Pluggable component factories
# ----------------------------------------

# TODO: Default for testing it smaller
def get_default_embedding() -> Any:
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(model_name=model_name)


# TODO: Replace with better chunking mechanism
def get_default_text_splitter() -> Any:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=int(os.getenv("CHUNK_SIZE", "300")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
    )


def get_default_vectordb(embedding: Optional[Any] = None) -> VectorDBInterface:
    if embedding is None:
        embedding = get_default_embedding()
    backend = os.getenv("VECTOR_DB", "chroma").lower()
    if backend == "pgvector":
        print("Using PGVector DB")
        db = PGVectorDBInstance(
            embedding_func=embedding,
            collection_name=os.getenv("PGVECTOR_COLLECTION", "my_documents"),
        )
        db.set_connection_string(
            user=os.getenv("PGVECTOR_USER", "postgres"),
            password=os.getenv("PGVECTOR_PASSWORD", "postgres"),
            host=os.getenv("PGVECTOR_HOST", "localhost"),
            dbname=os.getenv("PGVECTOR_DB", "postgres"),
            port=int(os.getenv("PGVECTOR_PORT", "5432")),
        )
        return db
    if backend == "chroma":
        print("Using ChromaDB")
        return ChromaDBInstance(
            embedding_func=embedding,
            persist_directory=os.getenv("CHROMA_DIR", "./chroma_db"),
        )
    raise ValueError(f"Unknown VECTOR_DB backend: {backend!r}")


# Process-level cache so the embedding model isn't reloaded per request
_default_vectordb: Optional[VectorDBInterface] = None


def _cached_default_vectordb() -> VectorDBInterface:
    global _default_vectordb
    if _default_vectordb is None:
        _default_vectordb = get_default_vectordb()
    return _default_vectordb


# ----------------------------------------
# FastAPI dependencies (overridable in tests)
# ----------------------------------------

def get_vectordb_dep() -> VectorDBInterface:
    return _cached_default_vectordb()


def get_text_splitter_dep() -> Any:
    return get_default_text_splitter()


# ----------------------------------------
# Core indexing logic
# ----------------------------------------

def index_document(
    file_bytes: bytes,
    *,
    filename: str = "",
    vectordb: Optional[VectorDBInterface] = None,
    text_splitter: Any = None,
) -> bool:
    pages = extract_text_and_images_from_paper(BytesIO(file_bytes))

    if vectordb is None:
        vectordb = _cached_default_vectordb()
    
    doc_id = vectordb.store_pdf(filename=filename, file_bytes=file_bytes)

    pages = [
        Document(
            page_content=page.texts or "",
            metadata={
                "num_images": len(page.images),
                "filename": filename,
                "doc_id" : doc_id,
                "page_index": i,
            }
        )
        for i, page in enumerate(pages)
    ]
    if text_splitter is None:
        text_splitter = get_default_text_splitter()
    chunks = text_splitter.split_documents(pages)

    if vectordb is None:
        vectordb = _cached_default_vectordb()
    vectordb.index_documents(chunks)
    return True
