from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


# ----------------------------------------
# LLM calls 
# ----------------------------------------

class LLMCallRequest(BaseModel):
    text: str 
@app.post("/generate")
async def llm_call(request: LLMCallRequest) -> str:

    """
    Make a call to the LLM with the given input and parameters.

    :param text: The input text for the LLM.

    Returns:
        string: The response from the LLM.
    """
    raise NotImplementedError()

