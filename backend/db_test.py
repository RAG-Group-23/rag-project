from pathlib import Path
import base64
import os
import tempfile

import requests
from fastapi.testclient import TestClient
from langchain_huggingface import HuggingFaceEmbeddings

from main import app  
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
            "/documents",
            json={
                "raw_document": payload,
                "filename": "paper.pdf",
                "session_id" : "testy"
            },
        )
        assert r.status_code == 200, r.text
        doc_id = r.json()
        assert isinstance(doc_id, str) and len(doc_id) > 0

        vectorstore = test_vectordb.get_vectorstore()
        retriever = vectorstore.as_retriever()
        docs = retriever.invoke("What is attention?")
        expected = "An example of the attention mechanism following long-distance dependencies in the"
        total = "".join(doc.page_content for doc in docs)
        assert expected in total
    finally:
        app.dependency_overrides.clear()
        Path(PDF_FILE).unlink(missing_ok=True)


def test_search_embedding_returns_relevant_chunks():
    pdf_bytes = _ensure_pdf_bytes()
    payload = base64.b64encode(pdf_bytes).decode("ascii")

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
            "/documents",
            json={
                "raw_document": payload,
                "filename": "paper.pdf",
                "session_id": "testy"
            },
        )
        assert r.status_code == 200, r.text
        doc_id = r.json()

        r = client.post(
            "/search/embedding",
            json={
                "document_ids": [doc_id],
                "query": "What is attention?",
            },
        )
        assert r.status_code == 200, r.text

        chunks = r.json()
        assert len(chunks) > 0, "Expected at least one chunk"
        total = "".join(chunks)
        assert "attention" in total.lower()

    finally:
        app.dependency_overrides.clear()
        Path(PDF_FILE).unlink(missing_ok=True)


if __name__ == "__main__":
    test_add_document_indexes_and_retrieves()
    test_search_embedding_returns_relevant_chunks()
    print("OK")
