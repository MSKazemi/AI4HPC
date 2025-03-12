# retrieval.py
import os
from typing import List

from dotenv import load_dotenv
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

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
        print(documents)
        answer = "\n".join([doc.page_content for doc in documents])
        return answer
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
llm = ChatOpenAI()



if __name__ == "__main__":
    # Initialize retriever with OpenAI API key
    retriever = DocumentRetriever()

    query = "What is infinband in HPC?"
    complete_answer = retriever.get_complete_answer(query)
    messages = [
    SystemMessage("You are AI assistant. Give me all answers in italian."),
    HumanMessage(content= f"context:\n{complete_answer}\n\n Question: {query}\n\n"),]
    res = llm.invoke(messages)
    print("🔍 Complete Answer:")
    print(complete_answer)
    print("🤖 AI Response:",res)


