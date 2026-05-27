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


## Chroma DB with lower model
Mainly used for fast local testing.
```python
VECTOR_DB=chroma LOAD_LLM=true LLM_MODEL=HuggingFaceTB/SmolLM2-360M-Instruct EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 python3 main.py
```
* Will use ChromaDB instead of PGVector
* If LOAD_LLM=true, will use the specified LLM. The one specified here are much weaker than the production one and can be run from CPU
    * Weaker models are also dumber, so do not expect responses directly related to documents; just a way to test interaction
* If LOAD_LLM=false, will use a MockLLM, that is, no LLM at all. Will just return the chunks as a formatted response.

## Chroma DB with higher model
```
VECTOR_DB=chroma LOAD_LLM=true python3 main.py
```

## PG Vector
To check if DB contains right data; use the following commands:

**Setup**
```python
import psycopg2
HOST = "nv-service-b052326c6f1b0774445ab8188cbe228a"
conn = psycopg2.connect(
    dbname="nuvolos",
    user="nuvolos",
    host=HOST,
    port=5432,
    password='nuvolos'
)
cur = conn.cursor()
```
* `cur` is an uncommited transaction; mainly used to check data.

**Tables**
```py
cur.execute("""
    SELECT tablename
    FROM pg_tables
    WHERE schemaname = 'public';
""")

tables = cur.fetchall()
print(tables)
```
```
[('pdf_documents',), ('conversations',), ('chunks',)]
```
* `pdf_documents` stores the binary data of PDFs
* `conversation` stores the user/assistent interactions
* `chunks` stores the embeddings


**Deletion**
```py
cur.execute("""
    DROP TABLE IF EXISTS pdf_documents CASCADE;
    DROP TABLE IF EXISTS conversation CASCADE;
    DROP TABLE IF EXISTS chunks CASCADE;
""")
conn.commit()
```


**Viewing PDF Files**
```py
cur.execute("SELECT * FROM pdf_documents;")
data = cur.fetchall()
for d in data:
    print(d)
```
```
('62770a86-db91-4e36-b17a-4826f4dddc4d', 'MT_with_ChatGPT.pdf', <memory at 0x7f3abbbd5d80>, datetime.datetime(2026, 5, 6, 9, 9, 10, 608126, tzinfo=datetime.timezone.utc))
```
* LangChain's PGVector handling does not support PDF storage by default
* We store IDs of the PDFDocument in the metadata of the embedding
* We store the PDF as byte arrays in a table
* So both PDFs & Embeddings are stored differently.


**Embeddings**
```py
cur.execute("""
    SELECT langchain_id, langchain_metadata 
    FROM chunks LIMIT 1""")
items = cur.fetchall()
for item in items:
    print(item)
```
```
('b704aa1a-6809-44bd-8496-a5fe89b7138f', {'document_id': '896bbe50ff4b0c0c319ac2f635117bb892c6fee9145b26d8335b7e68a8bd2ded', 'section': '**A Fine-Grained Analysis of BERTScore**', 'section_level': 2, 'section_index': 0, 'page_start': 1, 'page_end': 1, 'num_images': 0, 'local_chunk_index': 0, 'page': 1, 'chunk_id': '896bbe50ff4b0c0c319ac2f635117bb892c6fee9145b26d8335b7e68a8bd2ded_chunk_0', 'chunk_index': 0, 'filename': 'BERTScore_Experiments.pdf'})
```

## Local Testing of PGVector
Create `docker-compose.yml` with:
```
services:
  pgvector:
    image: pgvector/pgvector:pg17
    container_name: pgvector_dev
    environment:
      POSTGRES_USER: dev
      POSTGRES_PASSWORD: dev
      POSTGRES_DB: ragdb
    ports:
      - "5432:5432"
    volumes:
      - pgvector_data:/var/lib/postgresql/data

volumes:
  pgvector_data:
```

Create a `.env` with:
```
VECTOR_DB=pgvector
LOAD_LLM=false
LLM_MODEL=HuggingFaceTB/SmolLM2-360M-Instruct
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
DB_HOST=localhost
DB_PORT=5432
DB_NAME=ragdb
DB_USER=dev
DB_PASSWORD=dev
```

cd into the `rag-project/backend` and run:
```
env $(cat .env | xargs) python main.py
```
Allows you to check if PGVector related functionality works without using Nuvolus.


# Troubleshooting
## PGVector DB Start Up Issues
In some cases, PGVector DB may not have terminated successfully which may cause `/dhlib/postmaster.pid` to remain. At the next start up, PGVector DB refuses to start as that file still exists. To fix this, run this inside the shell:
```sh
rm /dhlib/postmaster.pid
```
Then restart the application.
