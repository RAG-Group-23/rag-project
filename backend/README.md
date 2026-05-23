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


## Testing Locally
```python
VECTOR_DB=chroma LOAD_MODELS=true LLM_MODEL=HuggingFaceTB/SmolLM2-360M-Instruct EMBEDDER_MODEL=sentence-transformers/all-MiniLM-L6-v2 python3 main.py
or 
VECTOR_DB=chroma LOAD_MODELS=true python3 main.py
```
* Will use ChromaDB instead of PGVector
* If LOAD_MODELS=true, will use the specified models. The ones specified here are much weaker than the production one and can be run from CPU
    * Weaker models are also dumber, so do not expect responses directly related to documents; just a way to test interaction
* If LOAD_MODELS=false, will use a MockLLM, that is, no LLM at all. Will just return the chunks as a formatted response.

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

**Deletion**
```py
cur.execute("""
    DROP TABLE IF EXISTS langchain_pg_embedding CASCADE;
    DROP TABLE IF EXISTS langchain_pg_collection CASCADE;
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
cur.execute("SELECT * FROM langchain_pg_collection;")
print(cur.fetchall())
```
```
[('my_documents', None, 'e954ea3f-a0cc-466a-a9f6-4d737aecd77a')]
```


```py
cur.execute("SELECT uuid, cmetadata FROM langchain_pg_embedding;")
data = cur.fetchall()
print("Size:", len(data))
for i,d in enumerate(data):
    if i<=5:
        print(d)
```
```
[('pdf_documents',), ('langchain_pg_collection',), ('langchain_pg_embedding',)]
[('my_documents', None, 'e954ea3f-a0cc-466a-a9f6-4d737aecd77a')]
Size: 59
('a2377655-4e22-4098-b120-29ddbcb931c1', {'num_images': 5, 'filename': 'MT_with_ChatGPT.pdf', 'doc_id': '62770a86-db91-4e36-b17a-4826f4dddc4d', 'page_index': 0})
('be60079f-913c-41c6-b9e0-bae2258a0690', {'num_images': 5, 'filename': 'MT_with_ChatGPT.pdf', 'doc_id': '62770a86-db91-4e36-b17a-4826f4dddc4d', 'page_index': 0})
('01638e17-8423-4bf1-a265-4f09b88ec7af', {'num_images': 5, 'filename': 'MT_with_ChatGPT.pdf', 'doc_id': '62770a86-db91-4e36-b17a-4826f4dddc4d', 'page_index': 0})
('2c83fefe-5a01-46e6-b4d7-fb2247ca3014', {'num_images': 5, 'filename': 'MT_with_ChatGPT.pdf', 'doc_id': '62770a86-db91-4e36-b17a-4826f4dddc4d', 'page_index': 0})
('19e5d5c4-d34b-4944-aa9b-f6c77fee0630', {'num_images': 5, 'filename': 'MT_with_ChatGPT.pdf', 'doc_id': '62770a86-db91-4e36-b17a-4826f4dddc4d', 'page_index': 0})
('b7f0e67a-4f8f-4571-8f4b-2be7e4f97bf9', {'num_images': 4, 'filename': 'MT_with_ChatGPT.pdf', 'doc_id': '62770a86-db91-4e36-b17a-4826f4dddc4d', 'page_index': 1})
```

# Troubleshooting
## PGVector DB Start Up Issues
In some cases, PGVector DB may not have terminated successfully which may cause `/dhlib/postmaster.pid` to remain. At the next start up, PGVector DB refuses to start as that file still exists. To fix this, run this inside the shell:
```sh
rm /dhlib/postmaster.pid
```
Then restart the application.
