# retrieval.py
import os
import numpy as np
from dotenv import load_dotenv
import chromadb
from typing import List, Dict
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

class DocumentRetriever:
    """
    Retrieves relevant documents from ChromaDB based on similarity search.
    """

    def __init__(self, persist_directory="vectorstore/chroma_db", openai_api_key=openai_api_key, top_k=3):
        """
        Initializes ChromaDB retriever.

        :param persist_directory: str - Directory where ChromaDB stores vectors.
        :param openai_api_key: str - API key for OpenAI embeddings.
        :param top_k: int - Number of relevant results to return.
        """
        self.top_k = top_k
        self.persist_directory = persist_directory

        # Load OpenAI embeddings
        self.embedding_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

        # Load ChromaDB
        self.chroma_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model
        )

        # Create retriever instance
        self.retriever = self.chroma_store.as_retriever(search_kwargs={"k": self.top_k})

    def retrieve_documents(self, query: str) -> List[Document]:
        """
        Retrieves the most relevant documents for a given query.

        :param query: str - User's question.
        :return: List[Document] - List of relevant documents sorted by similarity.
        """
        return self.retriever.invoke(query)

    def get_complete_answer(self, query: str) -> str:
        """
        Retrieves the most relevant documents and constructs a complete answer.

        :param query: str - User's question.
        :return: str - Complete answer constructed from relevant documents.
        """
        documents = self.retrieve_documents(query)
        answer = "\n".join([doc.page_content for doc in documents])
        return answer

if __name__ == "__main__":
    # Initialize retriever with OpenAI API key
    retriever = DocumentRetriever()

    query = "What is userDB?"
    complete_answer = retriever.get_complete_answer(query)

    print("🔍 Complete Answer:")
    print(complete_answer)