# Back-end

## DB Service
The DB service manages all database-related tasks.
* [ ] File indexing:
    1. [ ] Text extraction
    2. [ ] Text chunking
    3. [ ] Chunk upload to database
* [ ] Implement keyword search
* [ ] Implement embedding search


## General Services
General backend services. The front-end only calls this service for all functions.
* [ ] Create/design general system boot sequence (Because we have multiple nuvolos VM this might take a some manual work)
* [ ] Context management
* [ ] Search functionality (combines keyword search and embedding search)
* [ ] Implement communication with LLM and DB services (each service probably needs to have a FastAPI server)


## ML Service
A service that manages all ML-related components (model management, inference).
* [ ] Decide on an LLM to use
* [ ] Implement LLM inference

* [ ] Decide on embedding model
* [ ] Implement embedding model inference


# Front-end
Web interface that allows the user to interact with the system.
* [ ] Implement chat functionality
* [ ] Implement communication between back-end and front-end

