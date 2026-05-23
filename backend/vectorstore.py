import os
import json
import psycopg2
from psycopg2 import pool as psycopg2_pool
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from langchain_community.vectorstores import VectorStore
import hashlib
from langchain_postgres import PGEngine, PGVectorStore
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever


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

    @abstractmethod
    def delete_session(self, session_id: str) -> None:
        """Delete all messages belonging to a session.

        Args:
            session_id: The session whose messages should be removed.

        Raises:
            KeyError: If no messages exist for the given session_id.
        """
        pass

    @abstractmethod
    def delete_document(self, doc_id: str) -> None:
        """Delete a stored PDF and its associated vector chunks.

        Args:
            doc_id: The document identifier returned by store_pdf.

        Raises:
            KeyError: If no document exists for the given doc_id.
        """
        pass
    
    @abstractmethod
    def as_retriever(self, doc_ids: list[str] | None = None, k: int = 4) -> VectorStoreRetriever:
        pass

# ---------------------------------------------------------------------------
# PostgreSQL / pgvector backend
# ---------------------------------------------------------------------------

class PGVectorDBInstance(VectorDBInterface):
    def __init__(
        self,
        embedding_func: Embeddings,
        collection_name: str,
        vector_size: int,
        pool_min_conn: int = 1,
        pool_max_conn: int = 10,
    ):
        self.embedding = embedding_func
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._pool_min = pool_min_conn
        self._pool_max = pool_max_conn

        self._connection_string: str | None = None       # SQLAlchemy URL
        self._raw_conn_string: str | None = None         # plain psycopg2 URL
        self._pool: psycopg2_pool.ThreadedConnectionPool | None = None

        self.engine: PGEngine | None = None
        self.vectorstore: PGVectorStore | None = None

        # Track which tables have already been verified so we only pay the
        # CREATE TABLE IF NOT EXISTS cost once per process lifetime.
        self._tables_ensured: set[str] = set()

    # ------------------------------------------------------------------
    # Connection setup
    # ------------------------------------------------------------------

    def set_connection_string(
        self,
        user: str,
        password: str,
        host: str,
        dbname: str,
        port: int = 5432,
    ) -> None:
        self._connection_string = (
            f"postgresql+psycopg://{user}:{password}@{host}:{port}/{dbname}"
        )
        self._raw_conn_string = (
            f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        )
        self.engine = PGEngine.from_connection_string(
            url=self._connection_string)
        self._pool = psycopg2_pool.ThreadedConnectionPool(
            self._pool_min,
            self._pool_max,
            self._raw_conn_string,
        )

    def _conn(self):
        """Context manager that borrows a connection from the pool."""
        assert self._pool is not None, "Call set_connection_string first."

        class _Ctx:
            def __init__(self, pool):
                self._pool = pool
                self._conn = None

            def __enter__(self):
                self._conn = self._pool.getconn()
                return self._conn

            def __exit__(self, exc_type, *_):
                if exc_type is None:
                    self._conn.commit()
                else:
                    self._conn.rollback()
                self._pool.putconn(self._conn)

        return _Ctx(self._pool)

    # ------------------------------------------------------------------
    # Vector table management
    # ------------------------------------------------------------------

    def init_vector_table(self) -> None:
        """Create the vector table if it doesn't exist. Safe to call at startup."""
        assert self.engine is not None, "Call set_connection_string first."
        if not self._vector_table_exists():
            print("INFO: Vector table does not exist, creating one…")
            self.engine.init_vectorstore_table(
                table_name=self.collection_name,
                vector_size=self.vector_size,
                content_column="content",
            )

    def _vector_table_exists(self) -> bool:
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT to_regclass(%s)",
                    (f"public.{self.collection_name}",),
                )
                return cur.fetchone()[0] is not None

    # ------------------------------------------------------------------
    # Lazy table creation (once per table per process)
    # ------------------------------------------------------------------

    def _ensure_documents_table(self) -> None:
        if "pdf_documents" in self._tables_ensured:
            return
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS pdf_documents (
                        document_id TEXT        PRIMARY KEY,
                        filename    TEXT        NOT NULL,
                        pdf         BYTEA       NOT NULL,
                        uploaded_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                """)
        self._tables_ensured.add("pdf_documents")

    def _ensure_conversations_table(self) -> None:
        if "conversations" in self._tables_ensured:
            return
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id         BIGSERIAL   PRIMARY KEY,
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
        self._tables_ensured.add("conversations")

    # ------------------------------------------------------------------
    # PDF storage
    # ------------------------------------------------------------------

    def store_pdf(self, filename: str, file_bytes: bytes) -> str:
        self._ensure_documents_table()
        doc_id = document_hash(file_bytes)
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO pdf_documents (document_id, filename, pdf)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (document_id) DO NOTHING
                    """,
                    (doc_id, filename, psycopg2.Binary(file_bytes)),
                )
        return doc_id

    def fetch_pdf(self, doc_id: str) -> tuple[str, bytes]:
        self._ensure_documents_table()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT filename, pdf FROM pdf_documents WHERE document_id = %s",
                    (doc_id,),
                )
                row = cur.fetchone()
        if row is None:
            raise KeyError(f"No document found for document_id={doc_id!r}")
        filename, pdf_bytes = row
        return filename, bytes(pdf_bytes)

    def delete_document(self, doc_id: str) -> None:
        """Delete a PDF and all its associated vector chunks."""
        self._ensure_documents_table()

        # Verify the document exists before doing anything.
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM pdf_documents WHERE document_id = %s",
                    (doc_id,),
                )
                if cur.fetchone() is None:
                    raise KeyError(
                        f"No document found for document_id={doc_id!r}")

        # Remove vector chunks that carry this document_id in their metadata.
        vs = self.get_vectorstore()
        vs.delete(filter={"document_id": doc_id})

        # Remove the raw PDF row.
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM pdf_documents WHERE document_id = %s",
                    (doc_id,),
                )

    # ------------------------------------------------------------------
    # Conversation storage
    # ------------------------------------------------------------------

    def store_message(self, session_id: str, role: str, message: str) -> None:
        self._ensure_conversations_table()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO conversations (session_id, role, message) VALUES (%s, %s, %s)",
                    (session_id, role, message),
                )

    def fetch_conversation(self, session_id: str) -> list[dict]:
        self._ensure_conversations_table()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT role, message, sent_at
                    FROM   conversations
                    WHERE  session_id = %s
                    ORDER  BY sent_at ASC
                    """,
                    (session_id,),
                )
                rows = cur.fetchall()
        return [
            {"role": role, "message": message, "sent_at": sent_at}
            for role, message, sent_at in rows
        ]

    def delete_session(self, session_id: str) -> None:
        """Delete all messages for a session."""
        self._ensure_conversations_table()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM conversations WHERE session_id = %s LIMIT 1",
                    (session_id,),
                )
                if cur.fetchone() is None:
                    raise KeyError(
                        f"No session found for session_id={session_id!r}")
                cur.execute(
                    "DELETE FROM conversations WHERE session_id = %s",
                    (session_id,),
                )

    def get_session_ids(self) -> list[str]:
        self._ensure_conversations_table()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT session_id FROM conversations ORDER BY session_id"
                )
                return [row[0] for row in cur.fetchall()]

    def get_document_ids(self) -> dict[str, str]:
        self._ensure_documents_table()
        with self._conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT document_id, filename FROM pdf_documents ORDER BY uploaded_at ASC"
                )
                return {doc_id: filename for doc_id, filename in cur.fetchall()}

    # ------------------------------------------------------------------
    # Vector store
    # ------------------------------------------------------------------

    def index_documents(self, documents: list) -> PGVectorStore:
        vs = self.get_vectorstore()
        vs.add_documents(documents)
        return vs

    def get_vectorstore(self) -> PGVectorStore:
        assert self.engine is not None, "Call set_connection_string first."
        if self.vectorstore is None:
            self.vectorstore = PGVectorStore.create_sync(
                engine=self.engine,
                table_name=self.collection_name,
                embedding_service=self.embedding,
            )
        return self.vectorstore

    def as_retriever(self, doc_ids: list[str] | None = None, k: int = 4) -> VectorStoreRetriever:
        vs = self.get_vectorstore()
        search_kwargs: dict = {"k": k}
        if doc_ids:
            search_kwargs["filter"] = {"document_id": {"$in": doc_ids}}
        return vs.as_retriever(search_kwargs=search_kwargs)


# ---------------------------------------------------------------------------
# Chroma / local filesystem backend
# ---------------------------------------------------------------------------

class ChromaDBInstance(VectorDBInterface):
    def __init__(
        self,
        embedding_func: Embeddings,
        persist_directory: str = "./chroma_db",
    ):
        print("INFO: Initialising ChromaDB")
        self.embedding = embedding_func
        self.persist_directory = persist_directory
        self.vectorstore: Chroma | None = None
        self._pdf_store_dir = os.path.join(persist_directory, "pdfs")
        self._conv_store_dir = os.path.join(persist_directory, "conversations")

    # ------------------------------------------------------------------
    # PDF storage
    # ------------------------------------------------------------------

    def store_pdf(self, filename: str, file_bytes: bytes) -> str:
        os.makedirs(self._pdf_store_dir, exist_ok=True)
        doc_id = document_hash(file_bytes)
        # File format: {doc_id}@{original_filename}
        dest = os.path.join(self._pdf_store_dir, f"{doc_id}@{filename}")
        if not os.path.exists(dest):
            with open(dest, "wb") as f:
                f.write(file_bytes)
        return doc_id

    def fetch_pdf(self, doc_id: str) -> tuple[str, bytes]:
        if not os.path.exists(self._pdf_store_dir):
            raise KeyError(f"No document found for document_id={doc_id!r}")
        # Files are stored as {doc_id}@{filename}
        matches = [
            f for f in os.listdir(self._pdf_store_dir)
            if f.startswith(f"{doc_id}@")
        ]
        if not matches:
            raise KeyError(f"No document found for document_id={doc_id!r}")
        filepath = os.path.join(self._pdf_store_dir, matches[0])
        # Strip the leading "{doc_id}@" prefix to recover the original filename.
        original_filename = matches[0][len(doc_id) + 1:]
        with open(filepath, "rb") as f:
            return original_filename, f.read()

    def delete_document(self, doc_id: str) -> None:
        """Delete a PDF file and all its associated vector chunks."""
        if not os.path.exists(self._pdf_store_dir):
            raise KeyError(f"No document found for document_id={doc_id!r}")
        matches = [
            f for f in os.listdir(self._pdf_store_dir)
            if f.startswith(f"{doc_id}@")
        ]
        if not matches:
            raise KeyError(f"No document found for document_id={doc_id!r}")

        # Remove vector chunks that carry this document_id in their metadata.
        vs = self.get_vectorstore()
        # Chroma supports metadata filters on delete
        vs.delete(filter={"document_id": doc_id})

        # Remove the raw PDF file.
        os.remove(os.path.join(self._pdf_store_dir, matches[0]))

    # ------------------------------------------------------------------
    # Conversation storage
    # ------------------------------------------------------------------

    def _session_path(self, session_id: str) -> str:
        os.makedirs(self._conv_store_dir, exist_ok=True)
        safe_id = session_id.replace("/", "_").replace("\\", "_")
        return os.path.join(self._conv_store_dir, f"{safe_id}.json")

    def store_message(self, session_id: str, role: str, message: str) -> None:
        path = self._session_path(session_id)
        messages: list[dict] = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                messages = json.load(f)
        messages.append(
            {
                "role": role,
                "message": message,
                "sent_at": datetime.now(timezone.utc).isoformat(),
            }
        )
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
            for entry in raw
        ]

    def delete_session(self, session_id: str) -> None:
        """Delete the JSON file that holds all messages for a session."""
        path = self._session_path(session_id)
        if not os.path.exists(path):
            raise KeyError(f"No session found for session_id={session_id!r}")
        os.remove(path)

    def get_session_ids(self) -> list[str]:
        if not os.path.exists(self._conv_store_dir):
            return []
        return [
            f.removesuffix(".json")
            for f in os.listdir(self._conv_store_dir)
            if f.endswith(".json")
        ]

    # ------------------------------------------------------------------
    # Vector store
    # ------------------------------------------------------------------

    def index_documents(self, documents: list) -> Chroma:
        vs = self.get_vectorstore()
        vs.add_documents(documents)
        return vs

    def get_document_ids(self) -> dict[str, str]:
        if not os.path.exists(self._pdf_store_dir):
            return {}
        result = {}
        for f in os.listdir(self._pdf_store_dir):
            if not f.endswith(".pdf"):
                continue
            # Files are stored as {doc_id}@{filename}
            at_idx = f.index("@")
            doc_id = f[:at_idx]
            filename = f[at_idx + 1:]
            result[doc_id] = filename
        return result

    def get_vectorstore(self) -> Chroma:
        if self.vectorstore is None:
            self.vectorstore = Chroma(
                embedding_function=self.embedding,
                persist_directory=self.persist_directory,
            )
        return self.vectorstore

    def as_retriever(self, doc_ids: list[str] | None = None, k: int = 4):
        vs = self.get_vectorstore()
        search_kwargs: dict = {"k": k}
        if doc_ids:
            search_kwargs["filter"] = {"document_id": {"$in": doc_ids}}
        return vs.as_retriever(search_kwargs=search_kwargs)
