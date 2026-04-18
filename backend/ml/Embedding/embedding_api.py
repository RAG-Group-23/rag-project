from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

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



