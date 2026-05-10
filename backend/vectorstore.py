# vectorstore.py
import os
import json
import uuid
import psycopg2
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from langchain_community.vectorstores import PGVector, Chroma
from langchain_community.vectorstores import VectorStore
import hashlib


def document_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class VectorDBInterface(ABC):
    @abstractmethod
    def index_documents(self, documents: list) -> VectorStore:
        pass

    @abstractmethod
    def get_vectorstore(self) -> VectorStore:
        pass

    @abstractmethod
    def store_pdf(self, filename: str, file_bytes: bytes) -> str:
        """Store a PDF and return a doc_id."""
        pass

    @abstractmethod
    def fetch_pdf(self, doc_id: str) -> tuple[str, bytes]:
        """Fetch a PDF by doc_id, returns (filename, bytes)."""
        pass

    @abstractmethod
    def store_message(self, session_id: str, role: str, message: str) -> None:
        """Append a message to a conversation session.

        Args:
            session_id: Unique identifier for the conversation session.
            role:       Speaker — typically 'user' or 'assistant'.
            message:    The message text.
        """
        pass

    @abstractmethod
    def fetch_conversation(self, session_id: str) -> list[dict]:
        """Return all messages for a session in ascending chronological order.

        Returns a list of dicts with keys:
            role     (str)
            message  (str)
            sent_at  (datetime, UTC)
        """
        pass

    @abstractmethod
    def get_document_ids(self) -> dict[str, str]:
        """Return a mapping of {doc_id: filename} for all stored documents."""
        pass
    

    @abstractmethod
    def get_session_ids(self) -> list[str]:
        """Return all session IDs that have at least one message."""
        pass
        

class PGVectorDBInstance(VectorDBInterface):
    def __init__(self, embedding_func, collection_name):
        self.connection_string = None
        self.embedding = embedding_func
        self.collection_name = collection_name
        self.vectorstore = None

    def set_connection_string(self, user: str, password: str, host: str, dbname: str, port: int = 5432):
        DRIVER = "psycopg2"
        self.connection_string = f"postgresql+{DRIVER}://{user}:{password}@{host}:{port}/{dbname}"

    def _pg_conn_string(self) -> str:
        """Plain psycopg2 connection string (no SQLAlchemy driver prefix)."""
        return self.connection_string.replace("postgresql+psycopg2://", "postgresql://")

    def _ensure_documents_table(self):
        with psycopg2.connect(self._pg_conn_string()) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS pdf_documents (
                        doc_id      TEXT PRIMARY KEY,
                        filename    TEXT NOT NULL,
                        pdf         BYTEA NOT NULL,
                        uploaded_at TIMESTAMPTZ DEFAULT now()
                    )
                """)

    def _ensure_conversations_table(self):
        with psycopg2.connect(self._pg_conn_string()) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id         BIGSERIAL PRIMARY KEY,
                        session_id TEXT        NOT NULL,
                        role       TEXT        NOT NULL,
                        message    TEXT        NOT NULL,
                        sent_at    TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS conversations_session_id_sent_at_idx
                        ON conversations (session_id, sent_at ASC)
                """)

    def store_pdf(self, filename: str, file_bytes: bytes) -> str:
        assert self.connection_string is not None, "Please set connection string!"
        self._ensure_documents_table()
        doc_id = document_hash(file_bytes)
        with psycopg2.connect(self._pg_conn_string()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO pdf_documents (doc_id, filename, pdf) VALUES (%s, %s, %s) ON CONFLICT (doc_id) DO NOTHING",
                    (doc_id, filename, psycopg2.Binary(file_bytes))
                )
        return doc_id

    def fetch_pdf(self, doc_id: str) -> tuple[str, bytes]:
        assert self.connection_string is not None, "Please set connection string!"
        with psycopg2.connect(self._pg_conn_string()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT filename, pdf FROM pdf_documents WHERE doc_id = %s",
                    (doc_id,)
                )
                row = cur.fetchone()
                if row is None:
                    raise KeyError(f"No document found for doc_id={doc_id}")
                filename, pdf_bytes = row
                return filename, bytes(pdf_bytes)

    def store_message(self, session_id: str, role: str, message: str) -> None:
        assert self.connection_string is not None, "Please set connection string!"
        self._ensure_conversations_table()
        with psycopg2.connect(self._pg_conn_string()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO conversations (session_id, role, message)
                    VALUES (%s, %s, %s)
                    """,
                    (session_id, role, message)
                )

    def fetch_conversation(self, session_id: str) -> list[dict]:
        assert self.connection_string is not None, "Please set connection string!"
        self._ensure_conversations_table()
        with psycopg2.connect(self._pg_conn_string()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT role, message, sent_at
                    FROM conversations
                    WHERE session_id = %s
                    ORDER BY sent_at ASC
                    """,
                    (session_id,)
                )
                return [
                    {"role": role, "message": message, "sent_at": sent_at}
                    for role, message, sent_at in cur.fetchall()
                ]

    def get_session_ids(self) -> list[str]:
        self._ensure_conversations_table()
        with psycopg2.connect(self._pg_conn_string()) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT DISTINCT session_id FROM conversations
                    ORDER BY session_id
                """)
                return [row[0] for row in cur.fetchall()]

    def index_documents(self, documents: list) -> VectorStore:
        assert self.connection_string is not None, "Please set connection string!"
        self.vectorstore = PGVector.from_documents(
            documents=documents,
            embedding=self.embedding,
            connection_string=self.connection_string,
            collection_name=self.collection_name
        )
        return self.vectorstore

    def get_document_ids(self) -> dict[str, str]:
        assert self.connection_string is not None, "Please set connection string!"
        self._ensure_documents_table()
        with psycopg2.connect(self._pg_conn_string()) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT doc_id, filename FROM pdf_documents ORDER BY uploaded_at ASC")
                return {doc_id: filename for doc_id, filename in cur.fetchall()}


    def get_vectorstore(self) -> VectorStore:
        assert self.connection_string is not None, "Please set connection string!"
        if self.vectorstore is None:
            self.vectorstore = PGVector(
                connection_string=self.connection_string,
                embedding_function=self.embedding,
                collection_name=self.collection_name
            )
        return self.vectorstore

    def as_retriever(self, doc_ids: list[str] | None = None, k: int = 4):
        vs = self.get_vectorstore()
        search_kwargs = {"k": k}
        if doc_ids is not None:
            search_kwargs["filter"] = {"doc_id": {"$in": doc_ids}}
        return vs.as_retriever(search_kwargs=search_kwargs)


class ChromaDBInstance(VectorDBInterface):
    def __init__(self, embedding_func, persist_directory: str = "./chroma_db"):
        print("Init ChromaDB")
        self.embedding = embedding_func
        self.persist_directory = persist_directory
        self.vectorstore = None
        self._pdf_store_dir = os.path.join(persist_directory, "pdfs")
        self._conv_store_dir = os.path.join(persist_directory, "conversations")

    def store_pdf(self, filename: str, file_bytes: bytes) -> str:
        os.makedirs(self._pdf_store_dir, exist_ok=True)
        doc_id = document_hash(file_bytes)
        dest = os.path.join(self._pdf_store_dir, f"{filename}@{doc_id}.pdf")
        if not os.path.exists(dest):
            with open(dest, "wb") as f:
                f.write(file_bytes)
        return doc_id

    def fetch_pdf(self, doc_id: str) -> tuple[str, bytes]:
        matches = [f for f in os.listdir(
            self._pdf_store_dir) if f.startswith(doc_id)]
        if not matches:
            raise KeyError(f"No document found for doc_id={doc_id}")
        filepath = os.path.join(self._pdf_store_dir, matches[0])
        _, original_filename = matches[0].split("@", 1)
        with open(filepath, "rb") as f:
            return original_filename, f.read()

    def _session_path(self, session_id: str) -> str:
        os.makedirs(self._conv_store_dir, exist_ok=True)
        # Sanitise session_id so it's safe as a filename
        safe_id = session_id.replace("/", "_").replace("\\", "_")
        return os.path.join(self._conv_store_dir, f"{safe_id}.json")

    def store_message(self, session_id: str, role: str, message: str) -> None:
        path = self._session_path(session_id)
        messages = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                messages = json.load(f)
        messages.append({
            "role": role,
            "message": message,
            "sent_at": datetime.now(timezone.utc).isoformat(),
        })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(messages, f, ensure_ascii=False, indent=2)

    def fetch_conversation(self, session_id: str) -> list[dict]:
        path = self._session_path(session_id)
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return [
            {
                "role": entry["role"],
                "message": entry["message"],
                "sent_at": datetime.fromisoformat(entry["sent_at"]),
            }
            for entry in raw  # already in insertion (ascending) order
        ]

    def get_session_ids(self) -> list[str]:
        if not os.path.exists(self._conv_store_dir):
            return []
        return [
            f.removesuffix(".json")
            for f in os.listdir(self._conv_store_dir)
            if f.endswith(".json")
        ]

    def index_documents(self, documents: list) -> VectorStore:
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding,
            persist_directory=self.persist_directory
        )
        return self.vectorstore

    def get_document_ids(self) -> dict[str, str]:
        if not os.path.exists(self._pdf_store_dir):
            return {}
        return {
            f.split("@")[1].removesuffix(".pdf"): f.split("@")[0]
            for f in os.listdir(self._pdf_store_dir)
            if f.endswith(".pdf")
        }

    def get_vectorstore(self) -> VectorStore:
        if self.vectorstore is None:
            self.vectorstore = Chroma(
                embedding_function=self.embedding,
                persist_directory=self.persist_directory
            )
        return self.vectorstore

    def as_retriever(self, doc_ids: list[str] | None = None, k: int = 4):
        vs = self.get_vectorstore()
        search_kwargs = {"k": k}
        if doc_ids is not None:
            search_kwargs["filter"] = {"doc_id": {"$in": doc_ids}}
        return vs.as_retriever(search_kwargs=search_kwargs)
