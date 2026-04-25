import base64
import os
import tempfile

import requests
from fastapi.testclient import TestClient
from langchain_huggingface import HuggingFaceEmbeddings


from db_api import app
from db import get_vectordb_dep, get_text_splitter_dep, get_default_text_splitter
from vectorstore import ChromaDBInstance


PDF_URL = "https://arxiv.org/pdf/1706.03762"
PDF_FILE = "paper.pdf"


def _ensure_pdf_bytes() -> bytes:
    if not os.path.exists(PDF_FILE):
        response = requests.get(PDF_URL)
        response.raise_for_status()
        with open(PDF_FILE, "wb") as f:
            f.write(response.content)
    with open(PDF_FILE, "rb") as f:
        return f.read()


def test_add_document_indexes_and_retrieves():
    pdf_bytes = _ensure_pdf_bytes()
    payload = base64.b64encode(pdf_bytes).decode("ascii")

    # Plug-and-play test stack: ChromaDB + MiniLM in an isolated temp directory
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    persist_dir = tempfile.mkdtemp(prefix="chroma_test_")
    test_vectordb = ChromaDBInstance(
        embedding_func=embedding,
        persist_directory=persist_dir,
    )

    app.dependency_overrides[get_vectordb_dep] = lambda: test_vectordb
    app.dependency_overrides[get_text_splitter_dep] = get_default_text_splitter

    try:
        client = TestClient(app)
        r = client.post(
            "/document",
            json={
                "raw_document": payload,
                "filename": "paper.pdf",
                "collection": "test-session",
            },
        )
        assert r.status_code == 200, r.text
        assert r.json() is True

        vectorstore = test_vectordb.get_vectorstore()
        retriever = vectorstore.as_retriever()
        docs = retriever.invoke("What is attention?")
        expected = "An example of the attention mechanism following long-distance dependencies in the"
        total = "".join(doc.page_content for doc in docs)
        assert expected in total
    finally:
        app.dependency_overrides.clear()


if __name__ == "__main__":
    test_add_document_indexes_and_retrieves()
    print("OK")