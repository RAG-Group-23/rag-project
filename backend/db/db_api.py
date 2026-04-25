import base64

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel

from db import index_document, get_vectordb_dep, get_text_splitter_dep

app = FastAPI()


#! ----------------------------------------
#! Database API endpoints
#! ----------------------------------------


#----------------------------------------
# Session management
#----------------------------------------

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

    :param session_id: The ID of the session to delete.

    Returns:
        bool: True if the session was deleted successfully, False otherwise.
    """
    raise NotImplementedError()


@app.get("/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    """
    Get session conversation and metadata.

    :param session_id: The ID of the session to retrieve.

    Returns:
        dict: A dictionary containing the session conversation and metadata.
    """
    raise NotImplementedError()


@app.get("/sessions")
def get_sessions() -> list:
    """
    Get a list of all sessions. It only contains metadata of each session, not the conversation. For displaying the session list on the frontend.

    Returns:
        list: A list of dictionaries containing metadata for each session.
    """
    raise NotImplementedError()

#----------------------------------------
# Conversation management
#----------------------------------------


class CreateConversation(BaseModel):
    message: str
@app.post("/session/{session_id}/conversation")
def create_conversation_for_session(session_id: str, request: CreateConversation) -> bool:
    """
    Creates a conversation mapped to session
    Should trigger a response from backend

    :param session_id: The ID of the session to which the message will be added.
    :param message: The user message to add.

    Returns:
        bool: True if the message was added successfully, False otherwise.
    """
    raise NotImplementedError()


class UpdateConversation(BaseModel):
    message: str
@app.put("/sessions/{session_id}/conversation")
def update_conversation_of_session(session_id: str, request: UpdateConversation) -> bool:
    """
    Add a user message to an existing session's conversation
    Should trigger a response from backend

    :param session_id: The ID of the session to which the message will be added.
    :param message: The user message to add.

    Returns:
        bool: True if the message was added successfully, False otherwise.
    """
    raise NotImplementedError()

@app.get("/sessions/{session_id}/conversation")
def get_conversation_of_session(session_id: str) -> dict:
    """
    Get the response message from the session. 
    It automatically adds the new response message to the session conversation.

    :param session_id: The ID of the session from which to retrieve the response.

    Returns:
        dict: The entire session's conversation with the new response message added.
    """
    raise NotImplementedError()


#----------------------------------------
# Raw document management
#----------------------------------------

class AddDocumentRequest(BaseModel):
    raw_document: str
    filename: str
    collection: str # session_id
@app.post("/documents")
def add_document(
    request: AddDocumentRequest,
    vectordb=Depends(get_vectordb_dep),
    text_splitter=Depends(get_text_splitter_dep),
) -> bool:
    """
    Uploads the document and indexes it.

    :param raw_document: The raw document content to add. (in base64 encoded)

    Returns:
        bool: True if the document was added successfully, False otherwise.
    """
    try:
        file_bytes = base64.b64decode(request.raw_document)
    except (ValueError, base64.binascii.Error) as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 payload: {e}")
    return index_document(file_bytes, vectordb=vectordb, text_splitter=text_splitter)


@app.get("/documents/{document_id}")
def get_document(document_id: str) -> str:
    """
    Get the document content and metadata.

    :param document_id: The ID of the document to retrieve.

    Returns:
        dict: A dictionary containing the document content and metadata.
    """
    raise NotImplementedError()


@app.delete("/documents/{document_id}")
def delete_document(document_id: str) -> bool:
    """
    Delete an existing document. 

    :param document_id: The ID of the document to delete.

    Returns:
        bool: True if the document was deleted successfully, False otherwise.
    """
    raise NotImplementedError()


# ----------------------------------------
# Search management 
# ----------------------------------------

class SearchKeywordRequest(BaseModel):
    list_of_document_ids: list[str]
    query: str
@app.get("/search/keyword")
def search_keyword(request: SearchKeywordRequest) -> list:
    """
    Search for relevant chunks using BM25 algorithm.

    :param list_of_document_ids: A list of document IDs to search within.
    :param query: The search query.

    Returns:
        list: A list of relevant chunks sorted by relevance score.
    """
    raise NotImplementedError()


class SearchEmbeddingRequest(BaseModel):
    list_of_document_ids: list[str]
    query: str
@app.get("/search/embedding")
def search_embedding(request: SearchEmbeddingRequest) -> list:
    """
    Search for relevant chunks using embedding similarity.

    :param list_of_document_ids: A list of document IDs to search within.
    :param query: The search query.

    Returns:
        list: A list of relevant chunks sorted by relevance score.
    """
    raise NotImplementedError()

