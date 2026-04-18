from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


#! ---------------------------------------
#! This is an interface for the frontend
#! ---------------------------------------



# ---------------------------------------
# Session management
# ---------------------------------------

@app.post("/sessions")
async def create_new_session() -> str:
    """
    Create a new empty session for the user. 

    Returns:
        session_id: The ID of the newly created session.
    """
    raise NotImplementedError()

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str) -> bool:
    """
    Delete an existing session. 

    :param session_id: The ID of the session to delete.

    Returns:
        bool: True if the session was deleted successfully, False otherwise.
    """
    raise NotImplementedError()    
    
@app.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict:
    """
    Get session conversation and metadata.

    :param session_id: The ID of the session to retrieve.

    Returns:
        dict: A dictionary containing the session conversation and metadata.
    """
    raise NotImplementedError()  

@app.get("/sessions") 
async def get_sessions_list() -> list:
    """
    Get a list of all sessions. It only contains metadata of each session, not the conversation. For displaying the session list on the frontend.

    Returns:
        list: A list of dictionaries containing metadata for each session.
    """
    raise NotImplementedError()  

# ---------------------------------------
# Conversation management
# ---------------------------------------

class AddUserMessageRequest(BaseModel):
    message: str
@app.post("/sessions/{session_id}/user-message")
async def add_user_message_to_session(session_id: str, request: AddUserMessageRequest) -> bool:
    """
    Add a user message to an existing session. 

    :param session_id: The ID of the session to which the message will be added.
    :param message: The user message to add.

    Returns:
        bool: True if the message was added successfully, False otherwise.
    """
    raise NotImplementedError()

@app.get("/sessions/{session_id}/response")
async def get_response_for_session(session_id: str) -> dict:
    """
    Get the response message from the session. 
    It automatically adds the new response message to the session conversation.

    :param session_id: The ID of the session from which to retrieve the response.

    Returns:
        dict: A dictionary containing the response message and any relevant metadata.
    """
    raise NotImplementedError()


# ---------------------------------------
# Document management
# ---------------------------------------

class AddDocumentRequest(BaseModel):
    raw_document: str
@app.post("/documents")
async def add_document(request: AddDocumentRequest) -> bool:
    """
    Uploads the document and indexes it. 

    :param document: The document to add.

    Returns:
        bool: True if the document was added successfully, False otherwise.
    """
    raise NotImplementedError()

@app.get("/documents/{document_id}")
async def get_document(document_id: str) -> dict:
    """
    Get the document content and metadata.

    :param document_id: The ID of the document to retrieve.

    Returns:
        dict: A dictionary containing the document content and metadata.
    """
    raise NotImplementedError()

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str) -> bool:
    """
    Delete an existing document. 

    :param document_id: The ID of the document to delete.

    Returns:
        bool: True if the document was deleted successfully, False otherwise.
    """
    raise NotImplementedError()

