from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_community.vectorstores import Chroma
from abc import ABC, abstractmethod


class VectorDBInterface(ABC):
    @abstractmethod
    def index_documents(self, documents: list):
        pass

    @abstractmethod
    def get_vectorstore(self):
        pass


class PGVectorDBInstance(VectorDBInterface):
    def __init__(self, embedding_func, collection_name):
        self.connection_string = None
        self.embedding = embedding_func
        self.collection_name = collection_name
        self.vectorstore = None

    def set_connection_string(self, user: str, password: str, host: str, dbname: str, port: int = 5432):
        DRIVER = "psycopg2"
        """
        Build a SQLAlchemy/Postgres connection string.
        """
        self.connection_string = f"postgresql+{DRIVER}://{user}:{password}@{host}:{port}/{dbname}"

    def index_documents(self, documents: list):
        assert self.connection_string is not None, 'Please set connection string!'
        self.vectorstore = PGVector.from_documents(
            documents=documents,
            embedding=self.embedding,
            connection_string=self.connection_string,
            collection_name=self.collection_name)
        return self.vectorstore

    def get_vectorstore(self):
        assert self.connection_string is not None, 'Please set connection string!'
        if self.vectorstore is None:
            self.vectorstore = PGVector(
                connection_string=self.connection_string,
                embedding_function=self.embedding,
                collection_name=self.collection_name
            )
        return self.vectorstore


class ChromaDBInstance(VectorDBInterface):
    def __init__(self, embedding_func, persist_directory: str = "./chroma_db"):
        self.embedding = embedding_func
        self.persist_directory = persist_directory
        self.vectorstore = None

    def index_documents(self, documents: list):
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding,
            persist_directory=self.persist_directory
        )
        #self.vectorstore.persist()
        return self.vectorstore

    def get_vectorstore(self):
        if self.vectorstore is None:
            self.vectorstore = Chroma(
                embedding_function=self.embedding,
                persist_directory=self.persist_directory
            )
        return self.vectorstore