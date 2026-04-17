from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


#! ----------------------------------------
#! ML service API endpoints 
#! ----------------------------------------


# ----------------------------------------
# LLM calls 
# ----------------------------------------

class LLMCallRequest(BaseModel):
    context: dict
@app.post("/llm")
async def llm_call(request: LLMCallRequest) -> str:

    """
    Make a call to the LLM with the given input and parameters.

    :param context: A dictionary of the conversation in a session.

    Returns:
        string: The response from the LLM.
    """
    raise NotImplementedError()


# ----------------------------------------
# Embedding calls (bi-encoder)
# ----------------------------------------

class GetEmbeddingsRequest(BaseModel):
    input: str
@app.post("/embeddings")
async def get_embeddings(request: GetEmbeddingsRequest) -> list:
    """
    Get the embeddings for the given input.

    :param input: The input text.

    Returns:
        np.ndarray: A list of embeddings.
    """
    raise NotImplementedError() 



