"""
RAG Backend API
Combines Nuvolos pgvector infrastructure with the full RAG API surface.
"""
import base64
import os

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator, field_validator
from contextlib import asynccontextmanager


from db import (
    index_document,
    retrieve_documents,
    get_vectordb_dep,
    get_text_splitter_dep,
    VectorDBInterface,
    store_message,
    fetch_conversation,
    get_document_ids,
    get_session_ids,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    if os.getenv("VECTOR_DB", "chroma") == "pgvector":
        from db import init_db
        init_db()
    yield

app = FastAPI(title="RAG Backend API", lifespan=lifespan)

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
# Load models 
# ----------------------------------------
if os.getenv("LOAD_MODELS", "false").lower() == "true":
    print("Loading models")
    from ml import LLM, Embedder
    llm = LLM("google/gemma-3-4b-it")
    embedder = Embedder("Qwen/Qwen3-Embedding-4B")
else:
    print("Skipping models loading")
    llm = None
    embedder = None


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
def get_sessions(vectordb=Depends(get_vectordb_dep)) -> list[str]:
    try:
        return get_session_ids(vectordb=vectordb)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error retrieving sessions: {e}")


# ----------------------------------------
# Conversation management
# ----------------------------------------
class CreateOrUpdateConversation(BaseModel):
    message: str
    role: str
    doc_ids: list[str]


class ConversationEntry(BaseModel):
    role: str
    message: str
    sent_at: str  # ISO-8601 UTC string


def _append_message(session_id: str, request: CreateOrUpdateConversation, vectordb: VectorDBInterface | None = None) -> str:
    # prompt = request.message  
    # if request.role == "user" and request.doc_ids:
    #     chunks = retrieve_documents(
    #         query=request.message,
    #         doc_ids=request.doc_ids,
    #         vectordb=vectordb,
    #     )
        
    #     # TODO for @MartynasKucys: Replace the following with prompt_builder & generator and return generator output
    #     context = '\n'.join([c.page_content[:100] for c in chunks])
    #     prompt = f'[MSG]:{request.message}\n[DOCS]:\n{context}'

    store_message(session_id=session_id, role=request.role,
                  message=request.message)
    # return prompt
    return request.message

@app.post("/sessions/{session_id}/conversation")
def create_conversation_for_session(
    session_id: str, 
    request: CreateOrUpdateConversation,
    vectordb=Depends(get_vectordb_dep),
) -> str:
    """
    Start or append to a conversation for a session.
 
    Returns:
        bool: True if stored successfully.
    """
    return _append_message(session_id, request, vectordb)


@app.put("/sessions/{session_id}/conversation")
def update_conversation_of_session(
    session_id: str, 
    request: CreateOrUpdateConversation,
    vectordb=Depends(get_vectordb_dep),
) -> str:
    """
    Append a message to an existing session's conversation.
 
    Returns:
        bool: True if stored successfully.
    """
    _append_message(session_id, request, vectordb)

    conversation = [{"role":message.role, "content":message.message} for message in get_conversation_for_session(session_id)]
    chunks = retrieve_documents(
        query=request.message,
        doc_ids=request.doc_ids,
        vectordb=vectordb,
    )

    response = llm.generate(conversation, chunks)
    response_request = CreateOrUpdateConversation(
        message=response,
        role="assistant",
        doc_ids=[])
    _append_message(session_id, response_request, vectordb)
    return response


@app.get("/sessions/{session_id}/conversation")
def get_conversation_for_session(
    session_id: str,
) -> list[ConversationEntry]:
    """
    Retrieve the full message history for a session in chronological order.
 
    Returns:
        List of messages with role, content, and timestamp.
    """
    messages = fetch_conversation(session_id=session_id)
    return [
        ConversationEntry(
            role=m["role"],
            message=m["message"],
            sent_at=m["sent_at"].isoformat(),
        )
        for m in messages
    ]



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

    @field_validator("raw_document")
    @classmethod
    def validate_base64(cls, v):
        if not v or not v.strip():
            raise ValueError("raw_document cannot be empty")
        return v

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, v):
        if not v or not v.strip():
            raise ValueError("filename cannot be empty")
        return v


@app.post("/documents")
def add_document(
    request: AddDocumentRequest,
    vectordb=Depends(get_vectordb_dep),
    text_splitter=Depends(get_text_splitter_dep),
) -> str:
    """
    Decode, store, and index a base64-encoded PDF.

    Returns:
        str: id of the indexed document
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


@app.get("/documents")
def get_documents(vectordb=Depends(get_vectordb_dep)) -> dict[str, str]:
    """
    Retrieve document ids of all indexed documents
    Returns:
        dict: Document id and associated filenames
    """
    try:
        return get_document_ids(vectordb=vectordb)
    except Exception as e:
        print("Error", e)
        raise HTTPException(
            status_code=500, detail=f"Error retrieving documents: {e}")


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
def search_embedding(
    request: SearchEmbeddingRequest,
    vectordb: VectorDBInterface = Depends(get_vectordb_dep),
) -> list:
    chunks = retrieve_documents(
        query=request.query,
        doc_ids=request.list_of_document_ids,
        vectordb=vectordb,
    )

    for chunk in chunks:
        print(chunk)

    return [chunk.page_content for chunk in chunks]

# ----------------------------------------
# Entrypoint
# ----------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("BACKEND_PORT", "8500"))
    uvicorn.run(app, host="0.0.0.0", port=port)