# RAG Backend API

A FastAPI server that stores documents in PostgreSQL (with pgvector) and
exposes endpoints for adding, listing, and querying documents via vector
similarity search.

## Running

```bash
pip install -r requirements.txt
python main.py                   # starts on port 8500
# or
python start_backend.py          # daemonizes, saves PID for stop_backend.py
```

Interactive docs: http://localhost:8500/docs

## Configuration

All settings come from environment variables (with sensible defaults):

| Env var       | Default                                        | Purpose              |
|--------------|------------------------------------------------|----------------------|
| `DB_HOST`    | `nv-service-d54c9117d23473fa7f28948da0635011`  | PostgreSQL hostname  |
| `DB_PORT`    | `5432`                                         | PostgreSQL port      |
| `DB_NAME`    | `nuvolos`                                      | Database name        |
| `DB_USER`    | `nuvolos`                                      | Database user        |
| `DB_PASSWORD`| `nuvolos`                                      | Database password    |

The default `DB_HOST` is a Nuvolos-assigned internal hostname. On the Nuvolos
internal network every pod gets a hostname like `nv-service-<hash>`, which
other pods on the same subnet can resolve — but nothing outside can.

## Network position

This backend is **not** exposed to the internet. The frontend server
reverse-proxies API requests to it over the Nuvolos internal network:

```
Browser ──► Frontend (port 3000) ──► this backend (port 8500) ──► PostgreSQL
            public-facing              internal only                internal only
```

CORS is set to `allow_origins=["*"]` because the frontend proxy makes the
requests server-side, not from a browser origin.

