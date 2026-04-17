from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


#! ----------------------------------------
#! Database API endpoints
#! ----------------------------------------


#----------------------------------------
# Session management
#----------------------------------------

@app.post("/session")
def create_new_session() -> str:
    """
    Create a new empty session for the user. 

    Returns:
        session_id: The ID of the newly created session.
    """
    raise NotImplementedError()


@app.delete("/session/{session_id}")
def delete_session(session_id: str) -> bool:
    """
    Delete an existing session. 

    :param session_id: The ID of the session to delete.

    Returns:
        bool: True if the session was deleted successfully, False otherwise.
    """
    raise NotImplementedError()


@app.get("/session/{session_id}")
def get_session(session_id: str) -> dict:
    """
    Get session conversation and metadata.

    :param session_id: The ID of the session to retrieve.

    Returns:
        dict: A dictionary containing the session conversation and metadata.
    """
    raise NotImplementedError()


@app.get("/session-list")
def get_session_list() -> list:
    """
    Get a list of all sessions. It only contains metadata of each session, not the conversation. For displaying the session list on the frontend.

    Returns:
        list: A list of dictionaries containing metadata for each session.
    """
    raise NotImplementedError()


#----------------------------------------
# Conversation management
#----------------------------------------

class AddUserMessageRequest(BaseModel):
    message: str
@app.post("/session/{session_id}/user-message")
def add_user_message_to_session(session_id: str, request: AddUserMessageRequest) -> bool:
    """
    Add a user message to an existing session. 

    :param session_id: The ID of the session to which the message will be added.
    :param message: The user message to add.

    Returns:
        bool: True if the message was added successfully, False otherwise.
    """
    raise NotImplementedError()


@app.get("/session/{session_id}/response")
def get_response_from_session(session_id: str) -> dict:
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
@app.post("/document")
def add_document(request: AddDocumentRequest) -> bool:
    """
    Uploads the document and indexes it. 

    :param raw_document: The raw document content to add. (in base64 encoded)

    Returns:
        bool: True if the document was added successfully, False otherwise.
    """
    raise NotImplementedError()


@app.get("/document/{document_id}")
def get_document(document_id: str) -> str:
    """
    Get the document content and metadata.

    :param document_id: The ID of the document to retrieve.

    Returns:
        dict: A dictionary containing the document content and metadata.
    """
    raise NotImplementedError()


@app.delete("/document/{document_id}")
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

