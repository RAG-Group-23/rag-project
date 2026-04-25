# Database API
Everything related to the database managament (including vector db) is handled by this API

## Production
```sh
cd backend/db
VECTOR_DB=pgvector uvicorn db_api:app --reload --port 8000
```
Following environment variables have to be set:
```sh
EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B
VECTOR_DB=pgvector
# TODO: Should collections be global/session based?
PGVECTOR_COLLECTION=documents
PGVECTOR_USER=<USER>
PGVECTOR_PASSWORD=<PASSWORD>
PGVECTOR_HOST=<HOST>
PGVECTOR_DBNAME=<DBNAME>
PGVECTOR_PORT=5432
```

## Local Testing
```sh
cd backend/db
uvicorn db_api:app --reload --port 8000
```

## Tests
```sh
pytest db_api_test.py -v
```