# Database API
Everything related to the database managament (including vector db) is handled by this API

## Docs
```sh
$ uvicorn db_api:app --reload --port 8000 --host 0.0.0.0 
```
* We use `0.0.0.0` to make it reachable from other applications
* On http://localhost:8000/docs you can then view the endpoints of the DB API 

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

# Postgres DB
## Connecting
```python
import psycopg2
HOST = <HOST FROM PROVIDER>
conn = psycopg2.connect(
    dbname="postgres",
    user="<USER>",
    host=HOST,
    port=5432,
    password='<PASSWORD>'
)
cur = conn.cursor()
```
## Queries
### Langchain Collection
```python
cur.execute("""
    SELECT table_name 
    FROM information_schema.tables
    WHERE table_schema = 'public';
""")
print(cur.fetchall())
```
```sh
[('langchain_pg_embedding',), ('langchain_pg_collection',)]
```

```python
cur.execute("SELECT * FROM langchain_pg_collection;")
print(cur.fetchall())
```
```sh
[('documents', None, 'f6098f0b-7dc8-4eb2-a31c-210727cfe5c2')]
```

### Chunks / Documents
Check chunks (documents) in collection
```python
cur.execute("SELECT uuid, cmetadata FROM langchain_pg_embedding;")
data = cur.fetchall()
print("Size:", len(data))
for d in data:
    print(d)
```
```sh
Size: 396
('a7bf5c12-7100-4882-ba19-2ae2748e6835', {'num_images': 0})
('72593af0-63b5-4eed-a37d-50fdeb04cfd9', {'num_images': 0})
('b29f89f9-87b2-4bf9-aa13-3673cd231000', {'num_images': 0})
('8536980f-fcbf-47ff-a9cf-69e50936caa9', {'num_images': 0})
('b763ec16-38c8-42fa-89b5-3098fbe54398', {'num_images': 0})
...
```
```python
cur.execute("SELECT uuid, document FROM langchain_pg_embedding WHERE uuid='15b51904-9f99-4d2e-9b79-fcb2ccc4aa68';")
data = cur.fetchall()
print("Size:", len(data))
for d in data:
    print(d[1])
```
```
Size: 1
plicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor
of 1√dk
. Additive attention computes the compatibility function using a feed-forward network with
a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is
much faster and more space-efficient in practice, since it can be implemented using highly optimized
matrix multiplication code.
While for small values of dk the two mechanisms perform similarly, additive attention outperforms
dot product attention without scaling for larger values of dk [3]. We suspect that for large values of
dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has
extremely small gradients 4. To counteract this effect, we scale the dot products by 1√dk
...
```



