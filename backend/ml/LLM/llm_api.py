from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


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

