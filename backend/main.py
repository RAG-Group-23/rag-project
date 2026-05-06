"""
RAG Backend API
Combines Nuvolos pgvector infrastructure with the full RAG API surface.
"""
import base64
import os

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator

from db import (
    index_document,
    get_vectordb_dep,
    get_text_splitter_dep,
)

app = FastAPI(title="RAG Backend API")

# Enable CORS for the frontend reverse proxy.
# The frontend server (not the browser) makes requests to this backend,
# so "*" is acceptable here — no browser ever talks to this host directly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------------------
# Lifecycle
# ----------------------------------------

@app.on_event("startup")
async def startup_event():
    """Verify DB connectivity and initialise schema on startup."""
    from db import init_db
    init_db()


# ----------------------------------------
# Health / root
# ----------------------------------------

@app.get("/")
async def root():
    return {"message": "RAG Backend API", "status": "running"}


@app.get("/health")
async def health():
    """Health check — verifies database connectivity."""
    from db import get_db_connection
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1;")
        cur.close()
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        raise HTTPException(
            status_code=503, detail=f"Database connection failed: {str(e)}")
    finally:
        if conn:
            conn.close()


# ----------------------------------------
# Session management
# ----------------------------------------

@app.post("/sessions")
def create_session() -> str:
    """
    Create a new empty session for the user.

    Returns:
        session_id: The ID of the newly created session.
    """
    raise NotImplementedError()


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str) -> bool:
    """
    Delete an existing session.

    Returns:
        bool: True if deleted successfully.
    """
    raise NotImplementedError()


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    """
    Get session conversation and metadata.

    Returns:
        dict: Session conversation and metadata.
    """
    raise NotImplementedError()


@app.get("/sessions")
def get_sessions() -> list:
    """
    List all sessions (metadata only, no conversation content).
    Intended for populating the session list on the frontend.

    Returns:
        list: Metadata for each session.
    """
    raise NotImplementedError()


# ----------------------------------------
# Conversation management
# ----------------------------------------

class CreateConversation(BaseModel):
    message: str


@app.post("/sessions/{session_id}/conversation")
def create_conversation_for_session(
    session_id: str, request: CreateConversation
) -> bool:
    """
    Create a conversation mapped to a session and trigger a backend response.

    Returns:
        bool: True if created successfully.
    """
    raise NotImplementedError()


class UpdateConversation(BaseModel):
    message: str


@app.put("/sessions/{session_id}/conversation")
def update_conversation_of_session(
    session_id: str, request: UpdateConversation
) -> bool:
    """
    Append a user message to an existing session's conversation
    and trigger a backend response.

    Returns:
        bool: True if updated successfully.
    """
    raise NotImplementedError()


@app.get("/sessions/{session_id}/conversation")
def get_conversation_of_session(session_id: str) -> dict:
    """
    Retrieve the full conversation for a session, including the latest response.

    Returns:
        dict: The complete conversation history.
    """
    raise NotImplementedError()


# ----------------------------------------
# Document management
# ----------------------------------------

class AddDocumentRequest(BaseModel):
    raw_document: str   # base64-encoded PDF bytes
    filename: str
    session_id: str

    @validator("raw_document")
    def validate_base64(cls, v):
        if not v or not v.strip():
            raise ValueError("raw_document cannot be empty")
        return v

    @validator("filename")
    def validate_filename(cls, v):
        if not v or not v.strip():
            raise ValueError("filename cannot be empty")
        return v


@app.post("/documents")
def add_document(
    request: AddDocumentRequest,
    vectordb=Depends(get_vectordb_dep),
    text_splitter=Depends(get_text_splitter_dep),
) -> bool:
    """
    Decode, store, and index a base64-encoded PDF.

    Returns:
        bool: True if indexed successfully.
    """
    try:
        file_bytes = base64.b64decode(request.raw_document)
    except (ValueError, base64.binascii.Error) as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid base64 payload: {e}")

    try:
        return index_document(
            file_bytes,
            filename=request.filename,
            vectordb=vectordb,
            text_splitter=text_splitter,
        )
    except Exception as e:
        print("Error", e)
        raise HTTPException(
            status_code=500, detail=f"Error indexing document: {e}")


@app.get("/documents/{document_id}")
def get_document(document_id: str) -> dict:
    """
    Retrieve document content and metadata by ID.

    Returns:
        dict: Document content and metadata.
    """
    raise NotImplementedError()


@app.delete("/documents/{document_id}")
def delete_document(document_id: str) -> bool:
    """
    Delete a document by ID.

    Returns:
        bool: True if deleted successfully.
    """
    raise NotImplementedError()


# ----------------------------------------
# Search
# ----------------------------------------

class SearchKeywordRequest(BaseModel):
    list_of_document_ids: list[str]
    query: str


@app.post("/search/keyword")
def search_keyword(request: SearchKeywordRequest) -> list:
    """
    BM25 keyword search over a specified set of documents.

    Returns:
        list: Relevant chunks sorted by relevance score.
    """
    raise NotImplementedError()


class SearchEmbeddingRequest(BaseModel):
    list_of_document_ids: list[str]
    query: str


@app.post("/search/embedding")
def search_embedding(request: SearchEmbeddingRequest) -> list:
    """
    Embedding similarity search over a specified set of documents.

    Returns:
        list: Relevant chunks sorted by similarity score.
    """
    raise NotImplementedError()


# ----------------------------------------
# Entrypoint
# ----------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("BACKEND_PORT", "8500"))
    uvicorn.run(app, host="0.0.0.0", port=port)
