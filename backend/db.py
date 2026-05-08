"""
db.py — Database initialisation, connection helpers, vectordb factories,
and core document-indexing logic.

Environment variables
---------------------
DB connection (Nuvolos pgvector host):
    DB_HOST         internal hostname of the PostgreSQL pod
    DB_PORT         default 5432
    DB_NAME         default "nuvolos"
    DB_USER         default "nuvolos"
    DB_PASSWORD     default "nuvolos"

Vector-DB backend selection:
    VECTOR_DB       "pgvector" (default on Nuvolos) | "chroma" (local dev)

pgvector-specific:
    PGVECTOR_COLLECTION   collection / table name (default "my_documents")
    -- connection reuses DB_HOST / DB_PORT / DB_NAME / DB_USER / DB_PASSWORD

ChromaDB-specific (local dev only):
    CHROMA_DIR      persist directory (default "./chroma_db")

Embedding / chunking:
    EMBEDDING_MODEL   HuggingFace model id
                      (default "sentence-transformers/all-MiniLM-L6-v2")
    CHUNK_SIZE        token chunk size   (default 300)
    CHUNK_OVERLAP     token chunk overlap (default 50)
"""

import os
from io import BytesIO
from typing import Any, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vectorstore import VectorDBInterface, PGVectorDBInstance, ChromaDBInstance
from text_extractor import extract_text_and_images_from_paper


# ----------------------------------------
# Nuvolos PostgreSQL connection settings
# ----------------------------------------

DB_HOST = os.getenv(
    "DB_HOST",     "nv-service-d54c9117d23473fa7f28948da0635011")
DB_PORT = os.getenv("DB_PORT",     "5432")
DB_NAME = os.getenv("DB_NAME",     "nuvolos")
DB_USER = os.getenv("DB_USER",     "nuvolos")
DB_PASSWORD = os.getenv("DB_PASSWORD", "nuvolos")


def get_db_connection():
    """Open a new psycopg2 connection to the Nuvolos PostgreSQL instance."""
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        cursor_factory=RealDictCursor,
    )


def init_db():
    """
    Ensure the pgvector extension and any application-level tables exist.
    Called once at FastAPI startup.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # pdf_documents table — stores raw PDFs uploaded by users.
        # The vectorstore layer (LangChain PGVector) manages its own tables
        # separately; this table is only for raw-file retrieval.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pdf_documents (
                doc_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                filename    TEXT NOT NULL,
                pdf         BYTEA NOT NULL,
                uploaded_at TIMESTAMPTZ DEFAULT now()
            );
        """)

        conn.commit()
        cur.close()
        conn.close()
        print("Database initialised successfully.")
    except Exception as e:
        print(f"Database initialisation error: {e}")


# ----------------------------------------
# Pluggable component factories
# ----------------------------------------

def get_default_embedding() -> Any:
    model_name = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    return HuggingFaceEmbeddings(model_name=model_name)


# TODO: Replace with a better chunking strategy for research papers
def get_default_text_splitter() -> Any:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=int(os.getenv("CHUNK_SIZE", "300")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
    )


def get_default_vectordb(embedding: Optional[Any] = None) -> VectorDBInterface:
    if embedding is None:
        embedding = get_default_embedding()

    backend = os.getenv("VECTOR_DB", "pgvector").lower()

    if backend == "pgvector":
        print("Using PGVector DB")
        db = PGVectorDBInstance(
            embedding_func=embedding,
            collection_name=os.getenv("PGVECTOR_COLLECTION", "my_documents"),
        )
        # Reuse the same Nuvolos DB credentials for the vector store
        db.set_connection_string(
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            dbname=DB_NAME,
            port=int(DB_PORT),
        )
        return db

    if backend == "chroma":
        print("Using ChromaDB (local dev)")
        return ChromaDBInstance(
            embedding_func=embedding,
            persist_directory=os.getenv("CHROMA_DIR", "./chroma_db"),
        )

    raise ValueError(f"Unknown VECTOR_DB backend: {backend!r}")


# Process-level cache — avoids reloading the embedding model on every request
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
) -> str:
    """
    Store a raw PDF and index its text chunks in the vector store.

    Steps
    -----
    1. Extract per-page text (and image counts) from the PDF bytes.
    2. Persist the raw PDF via vectordb.store_pdf() and obtain a doc_id.
    3. Wrap each page as a LangChain Document with metadata.
    4. Chunk the documents with the text splitter.
    5. Index the chunks into the vector store.

    Returns
    -------
    bool
        True on success; propagates exceptions on failure.
    """
    print("Test")
    if vectordb is None:
        vectordb = _cached_default_vectordb()
        print("Got VectorDB")
    if text_splitter is None:
        text_splitter = get_default_text_splitter()

    # 1. Extract text & image references from the PDF
    pages = extract_text_and_images_from_paper(BytesIO(file_bytes))

    # 2. Persist raw PDF
    doc_id = vectordb.store_pdf(filename=filename, file_bytes=file_bytes)

    # 3. Build LangChain Documents
    lc_pages = [
        Document(
            page_content=page.texts or "",
            metadata={
                "num_images": len(page.images),
                "filename": filename,
                "doc_id": doc_id,
                "page_index": i,
            },
        )
        for i, page in enumerate(pages)
    ]

    # 4. Chunk
    chunks = text_splitter.split_documents(lc_pages)

    # 5. Index
    vectordb.index_documents(chunks)

    return doc_id
