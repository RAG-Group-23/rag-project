from abc import ABC, abstractmethod

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector, Chroma

from langchain_huggingface import HuggingFaceEmbeddings


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

    def set_connection_string(
        self,
        user: str,
        password: str,
        host: str,
        dbname: str,
        port: int = 5432,
    ):
        driver = "psycopg2"
        self.connection_string = (
            f"postgresql+{driver}://{user}:{password}@{host}:{port}/{dbname}"
        )

    def index_documents(self, documents: list):
        assert self.connection_string is not None, "Please set connection string!"

        self.vectorstore = PGVector.from_documents(
            documents=documents,
            embedding=self.embedding,
            connection_string=self.connection_string,
            collection_name=self.collection_name,
        )
        return self.vectorstore

    def get_vectorstore(self):
        assert self.connection_string is not None, "Please set connection string!"

        if self.vectorstore is None:
            self.vectorstore = PGVector(
                connection_string=self.connection_string,
                embedding_function=self.embedding,
                collection_name=self.collection_name,
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
            persist_directory=self.persist_directory,
        )
        return self.vectorstore

    def get_vectorstore(self):
        if self.vectorstore is None:
            self.vectorstore = Chroma(
                embedding_function=self.embedding,
                persist_directory=self.persist_directory,
            )

        return self.vectorstore


if __name__ == "__main__":
    import requests

    from text_extractor import extract_text_and_images_from_paper
    from chunker import chunk_pages

    url = "https://arxiv.org/pdf/1706.03762"
    pdf_file = "paper.pdf"
    document_id = "attention_is_all_you_need"

    response = requests.get(url)
    response.raise_for_status()

    with open(pdf_file, "wb") as f:
        f.write(response.content)

    pages = extract_text_and_images_from_paper(pdf_file)

    splits = chunk_pages(
        pages=pages,
        document_id=document_id,
        chunk_size=1200,
        chunk_overlap=150,
    )

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # In production, replace this with PGVectorDBInstance.
    vector_db = ChromaDBInstance(
        embedding_func=embedding,
        persist_directory="./chroma_db_test"
    )

    vector_db.index_documents(splits)

    vectorstore = vector_db.get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    docs = retriever.invoke("How is scaled dot-product attention computed?")

    for doc in docs:
        print("-----")
        print(doc.metadata)
        print(doc.page_content)