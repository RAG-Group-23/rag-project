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
    PGVECTOR_COLLECTION   collection / table name (default "chunks")
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
import httpx
from typing import Any, Optional

import psycopg2
from psycopg2.extras import RealDictCursor

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings

from vectorstore import VectorDBInterface, PGVectorDBInstance, ChromaDBInstance
from text_extractor import extract_sections, Section
from chunker import chunk_sections


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


def init_pgvector_db():
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
                document_id      TEXT PRIMARY KEY,
                filename    TEXT NOT NULL,
                pdf         BYTEA NOT NULL,
                uploaded_at TIMESTAMPTZ DEFAULT now()
            );
        """)

        # conversations table — stores chat history keyed by session_id.
        # Ordered by sent_at; BIGSERIAL id is a tie-breaker for same-ms inserts.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id         BIGSERIAL PRIMARY KEY,
                session_id TEXT        NOT NULL,
                role       TEXT        NOT NULL,
                message    TEXT        NOT NULL,
                sent_at    TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS conversations_session_id_sent_at_idx
                ON conversations (session_id, sent_at ASC);
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

def get_embedding_and_size() -> tuple[Embeddings, int]:
    model_name = os.getenv(
        "EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B"
    )
    print("Loading embedding model:", model_name)
    match model_name:
        case "sentence-transformers/all-MiniLM-L6-v2":
            return HuggingFaceEmbeddings(model_name=model_name), 384
        case "Qwen/Qwen3-Embedding-4B":
            return HuggingFaceEmbeddings(model_name=model_name), 2560
        case _:
            raise ValueError(f"Unsupported model name: {model_name}")

# TODO: Replace with a better chunking strategy for research papers
def get_default_text_splitter() -> Any:
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=int(os.getenv("CHUNK_SIZE", "300")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
    )


def get_vectordb(embedding: Optional[Embeddings] = None, embedding_size: Optional[int] = None) -> VectorDBInterface:
    if embedding is None:
        embedding, embedding_size = get_embedding_and_size()

    backend = os.getenv("VECTOR_DB", "pgvector").lower()

    if backend == "pgvector":
        print("Using PGVector DB")
        db = PGVectorDBInstance(
            embedding_func=embedding,
            collection_name=os.getenv("PGVECTOR_COLLECTION", "chunks"),
            vector_size=embedding_size
        )
        db.set_connection_string(
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            dbname=DB_NAME,
            port=int(DB_PORT),
        )
        db.init_vector_table()
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
        _default_vectordb = get_vectordb()
    return _default_vectordb


# ----------------------------------------
# FastAPI dependencies (overridable in tests)
# ----------------------------------------

def get_vectordb_dep() -> VectorDBInterface:
    return _cached_default_vectordb()

# ----------------------------------------
# Core indexing logic
# ----------------------------------------

def get_title_for_sections(sections) -> str:
    for section in sections:
        if section.title is not None:
            return section.title.replace("*", '').strip().replace(' ', '_')


def _resolve_filename(sections, provided: str, auto_name: bool) -> str:
    if auto_name or not provided.strip():
        inferred = get_title_for_sections(sections)
        return f"{inferred}.pdf" if inferred else (provided or "document.pdf")
    return provided


def _index_document_core(
    file_bytes: bytes,
    filename: str,
    sections: list[Section],
    *,
    vectordb: VectorDBInterface,
) -> str:
    doc_id = vectordb.store_pdf(filename=filename, file_bytes=file_bytes)
    chunks = chunk_sections(
        sections,
        document_id=doc_id,
        chunk_size=int(os.getenv("CHUNK_SIZE", "256")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "64")),
    )
    for chunk in chunks:
        chunk.metadata["filename"] = filename
    vectordb.index_documents(chunks)
    return doc_id


def index_document(
    file_bytes: bytes,
    *,
    filename: str = "",
    auto_name: bool = False,
    vectordb: Optional[VectorDBInterface] = None,
) -> str:
    if vectordb is None:
        vectordb = _cached_default_vectordb()
    sections = extract_sections(BytesIO(file_bytes))
    resolved = _resolve_filename(sections, filename, auto_name)
    return _index_document_core(file_bytes, resolved, sections, vectordb=vectordb)


def index_document_from_url(
    url: str,
    *,
    auto_name: bool = True,
    vectordb: Optional[VectorDBInterface] = None,
) -> str:
    if vectordb is None:
        vectordb = _cached_default_vectordb()

    response = httpx.get(url, follow_redirects=True, timeout=30)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "")
    if "application/pdf" not in content_type:
        raise ValueError(
            f"URL does not point to a PDF (Content-Type: {content_type})")

    file_bytes = response.content
    if not file_bytes.startswith(b"%PDF"):
        raise ValueError(
            "File does not appear to be a valid PDF (missing %PDF header)")

    url_filename = url.split("/")[-1].split("?")[0] or "document.pdf"
    sections = extract_sections(BytesIO(file_bytes))
    resolved = _resolve_filename(sections, url_filename, auto_name)
    return _index_document_core(file_bytes, resolved, sections, vectordb=vectordb)

def retrieve_documents(
    query: str,
    *,
    doc_ids: list[str] | None = None,
    k: int = 10,
    vectordb: VectorDBInterface | None = None,
) -> list[Document]:
    """
    Retrieve the top-k most relevant chunks for a query.

    Parameters
    ----------
    query   : the search string
    doc_ids : optional list of doc_ids to scope the search to specific documents
    k       : number of chunks to return (default 10)
    vectordb: injectable vectordb instance; falls back to the cached default

    Returns
    -------
    List of LangChain Documents with page_content and metadata
    (filename, doc_id, page_index, num_images).
    """
    if vectordb is None:
        vectordb = _cached_default_vectordb()

    retriever = vectordb.as_retriever(doc_ids=doc_ids, k=k)
    return retriever.invoke(query)


def get_document_ids(vectordb: VectorDBInterface | None = None) -> list[str]:
    if vectordb is None:
        vectordb = _cached_default_vectordb()
    return vectordb.get_document_ids()


def delete_session_db(session_id: str, vectordb: VectorDBInterface | None = None):
    if vectordb is None:
        vectordb = _cached_default_vectordb()
    return vectordb.delete_session(session_id=session_id)

def delete_document_db(document_id: str, vectordb: VectorDBInterface | None = None):
    if vectordb is None:
        vectordb = _cached_default_vectordb()
    return vectordb.delete_document(doc_id=document_id)

# ----------------------------------------
# Conversation history
# ----------------------------------------

def store_message(
    session_id: str,
    role: str,
    message: str,
    *,
    vectordb: VectorDBInterface | None = None,
) -> None:
    """
    Append a single message to a conversation session.

    Parameters
    ----------
    session_id : unique identifier for the chat session
    role       : speaker — 'user' or 'assistant'
    message    : message text
    vectordb   : injectable vectordb instance; falls back to the cached default
    """
    if vectordb is None:
        vectordb = _cached_default_vectordb()
    vectordb.store_message(session_id=session_id, role=role, message=message)


def fetch_conversation(
    session_id: str,
    *,
    vectordb: VectorDBInterface | None = None,
) -> list[dict]:
    """
    Return the full message history for a session in chronological order.

    Parameters
    ----------
    session_id : unique identifier for the chat session
    vectordb   : injectable vectordb instance; falls back to the cached default

    Returns
    -------
    List of dicts with keys: role (str), message (str), sent_at (datetime UTC).
    Returns an empty list if the session has no history.
    """
    if vectordb is None:
        vectordb = _cached_default_vectordb()
    return vectordb.fetch_conversation(session_id=session_id)


def get_session_ids(vectordb: VectorDBInterface) -> list[str]:
    return vectordb.get_session_ids()