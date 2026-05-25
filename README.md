# Research Paper RAG

A retrieval-augmented generation application for asking questions about uploaded
research papers. The project combines a FastAPI backend, a Streamlit chat
frontend, document chunking, vector search, and optional local LLM inference.

## Features

- Upload PDF documents through the Streamlit interface.
- Extract and chunk paper text for retrieval.
- Store embeddings in either PostgreSQL with pgvector or local ChromaDB.
- Ask document-scoped questions in a chat interface.
- Keep chat sessions and conversation history.
- Run quickly in mock-model mode for local development.

## Project Structure

```text
.
|-- backend/                 # FastAPI API, vector DB integration, RAG logic
|   |-- main.py              # API entrypoint
|   |-- db.py                # DB setup, document indexing, session helpers
|   |-- vectorstore.py       # pgvector and Chroma vector-store wrappers
|   |-- ml.py                # LLM and embedding model wrappers
|   `-- requirements.txt     # Backend Python dependencies
|-- frontend/                # Streamlit frontend
|   `-- frontend.py
|-- _docs/                   # Architecture notes and diagrams
`-- README.md
```

## Architecture

```text
Browser -> Streamlit frontend -> FastAPI backend -> Vector store / database
                                      |
                                      v
                                LLM response
```

The backend exposes endpoints for documents, sessions, conversations, and
embedding search. The frontend calls those endpoints and provides the document
upload and chat experience.

## Requirements

- Python 3.10+
- A virtual environment is recommended.
- For local development: ChromaDB can be used without PostgreSQL.
- For Nuvolos or production-style deployment: PostgreSQL with pgvector.
- Optional GPU support is useful when `LOAD_MODELS=true`.

## Setup

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install backend dependencies:

```bash
pip install -r backend/requirements.txt
```

Install frontend dependencies if they are not already available:

```bash
pip install streamlit requests
```

## Run Locally

For a lightweight local run, use ChromaDB and skip loading the real models:

```powershell
cd backend
$env:VECTOR_DB = "chroma"
$env:LOAD_MODELS = "false"
python main.py
```

The backend starts on `http://localhost:8500`.

In a second terminal, start the frontend:

```bash
cd frontend
streamlit run frontend.py
```

The frontend defaults to calling `http://127.0.0.1:8500`.

## Running With Real Models

Set `LOAD_MODELS=true` to load the model wrappers in `backend/ml.py`:

```powershell
$env:LOAD_MODELS = "true"
python backend/main.py
```

By default, the backend uses:

- LLM: `google/gemma-3-4b-it`
- Embedding model: `Qwen/Qwen3-Embedding-4B` when model loading is enabled
- Mock LLM and mock embedder when `LOAD_MODELS=false`

If local model files are available, set `MODEL_ROOT` so the backend can find
them:

```powershell
$env:MODEL_ROOT = "C:\path\to\models"
```

## Configuration

The backend is configured through environment variables.

| Variable | Default | Description |
| --- | --- | --- |
| `BACKEND_PORT` | `8500` | FastAPI server port |
| `LOAD_MODELS` | `false` | Load real LLM and embedding models |
| `VECTOR_DB` | `chroma` in `main.py`, `pgvector` in DB factory | Vector store backend |
| `CHROMA_DIR` | `./chroma_db` | Local Chroma persistence directory |
| `PGVECTOR_COLLECTION` | `chunks` | pgvector collection name |
| `DB_HOST` | Nuvolos internal host | PostgreSQL host |
| `DB_PORT` | `5432` | PostgreSQL port |
| `DB_NAME` | `nuvolos` | PostgreSQL database |
| `DB_USER` | `nuvolos` | PostgreSQL user |
| `DB_PASSWORD` | `nuvolos` | PostgreSQL password |
| `MODEL_ROOT` | `/files` | Root directory for local model files |
| `HOST_IP` | unset | Frontend override for backend host |
| `CHUNK_SIZE` | `300` | Token chunk size |
| `CHUNK_OVERLAP` | `50` | Token chunk overlap |

For Nuvolos deployment, the backend can use the provided internal PostgreSQL
host and pgvector tables. For local development, `VECTOR_DB=chroma` is the
simplest option.

## API Overview

Once the backend is running, interactive API docs are available at:

```text
http://localhost:8500/docs
```

Important endpoints:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Check backend and database connectivity |
| `POST` | `/sessions` | Create a chat session |
| `GET` | `/sessions` | List sessions |
| `GET` | `/sessions/{session_id}/conversation` | Fetch chat history |
| `PUT` | `/sessions/{session_id}/conversation` | Send a message and receive an answer |
| `POST` | `/documents` | Upload and index a base64-encoded PDF |
| `GET` | `/documents` | List indexed documents |
| `POST` | `/search/embedding` | Retrieve relevant chunks for a query |

Some endpoints are currently placeholders and raise `NotImplementedError`,
including document deletion, document retrieval by ID, session deletion, and
keyword search.

## Development Notes

- `LOAD_MODELS=false` is useful for testing API and frontend flow without
  downloading or loading large models.
- `VECTOR_DB=chroma` stores local vector data in `CHROMA_DIR`.
- `VECTOR_DB=pgvector` expects PostgreSQL credentials and pgvector support.
- Uploaded PDFs are stored separately from their vector embeddings.
- The frontend supports multiple chat sessions and document selection.

## Troubleshooting

If the frontend cannot reach the backend, check that:

- The backend is running on port `8500`.
- `HOST_IP` is set when the frontend needs to call a remote backend.
- `VECTOR_DB=chroma` is set for local development without PostgreSQL.

If pgvector startup fails on Nuvolos because PostgreSQL did not shut down
cleanly, remove the stale PostgreSQL PID file from the shell:

```bash
rm /dhlib/postmaster.pid
```

Then restart the backend.
