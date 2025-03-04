# store.py

import os
from typing import Dict, List

import chromadb
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


class ChromaVectorStore:
    """
    A class to manage vector storage using ChromaDB.
    """

    def __init__(self, persist_directory="vectorstore/chroma_db", openai_api_key=None):
        """
        Initializes ChromaDB and OpenAI Embeddings.

        :param persist_directory: str - Directory where the ChromaDB will store vectors.
        :param openai_api_key: str - API key for OpenAI (if using OpenAI embeddings).
        """
        self.persist_directory = persist_directory
        os.makedirs(self.persist_directory, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=self.persist_directory)

        # Load OpenAI embeddings
        self.embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

        # Initialize Chroma vector store
        self.chroma_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model
        )

    def index_documents(self, texts: List[str], metadata: List[Dict[str, str]]):
        """
        Stores text embeddings in ChromaDB.

        :param texts: List[str] - List of text chunks.
        :param metadata: List[Dict[str, str]] - Metadata for each chunk (e.g., source file).

        :return: None
        """
        documents = [Document(page_content=text, metadata=meta) for text, meta in zip(texts, metadata)]
        self.chroma_store.add_documents(documents)
        print(f"✅ Indexed {len(texts)} documents into ChromaDB.")

    def retrieve_similar(self, query: str, top_k: int = 3) -> List[Document]:
        """
        Retrieves the most relevant documents from ChromaDB.

        :param query: str - User query.
        :param top_k: int - Number of relevant results to retrieve.

        :return: List[Document] - Retrieved documents sorted by similarity.
        """
        retriever = self.chroma_store.as_retriever(search_kwargs={"k": top_k})
        results = retriever.get_relevant_documents(query)
        for i, doc in enumerate(results):
            print(f"{i+1}. {doc.metadata.get('url', 'Unknown Source')} (Source: {doc.metadata.get('source', 'Unknown Source')})")
        return results

    def delete_all(self):
        """
        Deletes all stored embeddings in ChromaDB.

        :return: None
        """
        self.chroma_store.delete_collection()
        print("🗑️ ChromaDB store cleared.")

    def save(self):
        """
        Ensures the database is persisted.

        :return: None
        """
        self.chroma_store.persist()

