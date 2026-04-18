# Backend

## Endpoint inspections & testing
```
$ uvicorn backend.general.general_api:app --reload --port 8000
```
* On http://localhost:8000/docs you can then view the endpoints of that API (`general.general_api`).
* You can directly use the UI to test the endpoints (or use `curl`)