# embedder.py

import os
import numpy as np
import openai
from openai import OpenAI
from typing import List, Union
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

class TextEmbedder:
    """
    A class for generating embeddings using OpenAI's API or a local model (SentenceTransformers).
    """

    def __init__(self, model_name="text-embedding-ada-002", use_openai=True):
        """
        Initialize the embedder with a selected model.
        
        :param model_name: str - Model name for embeddings. Default is OpenAI's ada-002.
        :param use_openai: bool - Whether to use OpenAI API (True) or local SentenceTransformers (False).
        """
        self.use_openai = use_openai
        self.model_name = model_name

        if use_openai:
            self.api_key = os.getenv("OPENAI_API_KEY")  # Load API key from environment
            if not self.api_key:
                raise ValueError("OpenAI API key is missing. Set OPENAI_API_KEY in the environment.")
            self.client = OpenAI(api_key=self.api_key)  # Define self.client
        else:
            # Load local embedding model
            self.local_model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of text inputs.
        
        :param texts: List[str] - List of text chunks to embed.
        
        :return: List[List[float]] - List of embedding vectors.
        """
        if self.use_openai:
            return self._embed_with_openai(texts)
        else:
            return self._embed_with_local_model(texts)

    def _embed_with_openai(self, texts: List[str]) -> List[List[float]]:
        """
        Uses OpenAI's API to generate embeddings.
        
        :param texts: List[str] - List of text chunks.
        
        :return: List[List[float]] - List of embedding vectors.
        """
        try:
            response = self.client.embeddings.create(input=texts,
            model=self.model_name)
            return [item.embedding for item in response.data]

        except openai.OpenAIError as e:
            print(f"OpenAI API Error: {e}")
            return []

    def _embed_with_local_model(self, texts: List[str]) -> List[List[float]]:
        """
        Uses a local SentenceTransformers model to generate embeddings.
        
        :param texts: List[str] - List of text chunks.
        
        :return: List[List[float]] - List of embedding vectors.
        """
        return self.local_model.encode(texts, convert_to_numpy=True).tolist()


# TODO
# Using a Local Model (sentence-transformers)
# If you want to avoid OpenAI API costs and use a local embedding model, initialize it with:


# if __name__ == "__main__":
#     embedder = TextEmbedder(model_name="all-MiniLM-L6-v2", use_openai=False)  # Local model
#     texts = ["HPC clusters enable parallel computing.", "Slurm is a job scheduler."]
    
#     embeddings = embedder.embed_texts(texts)
#     print(embeddings[:2])
# Recommended Local Models:

# "all-MiniLM-L6-v2" → Fast & small (~80MB)
# "all-mpnet-base-v2" → More accurate but slower (~400MB)


if __name__ == "__main__":
    embedder = TextEmbedder(use_openai=True)  # Uses OpenAI API
    texts = ["HPC clusters enable parallel computing.", "Slurm is a job scheduler."]

    embeddings = embedder.embed_texts(texts)
    print(embeddings[:2])  # Print first two embeddings
