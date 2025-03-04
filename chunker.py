# chunker.py

from typing import List

import nltk
from langchain.text_splitter import RecursiveCharacterTextSplitter
from nltk.tokenize import sent_tokenize

# Ensure NLTK sentence tokenizer is available
nltk.download("punkt")

class TextChunker:
    """
    A class for chunking text into smaller segments for efficient processing and embedding.
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100, method: str = "recursive"):
        """
        Initializes the chunker with a specified method and parameters.

        :param chunk_size: int - The maximum token length of a chunk.
        :param chunk_overlap: int - The number of overlapping tokens between consecutive chunks.
        :param method: str - The chunking strategy to use ("recursive" or "sentence").
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.method = method

    def chunk_text(self, text: str) -> List[str]:
        """
        Splits text into smaller chunks based on the selected chunking method.

        :param text: str - The full document text.

        :return: List[str] - List of text chunks.
        """
        if self.method == "recursive":
            return self._recursive_chunk(text)
        elif self.method == "sentence":
            return self._sentence_chunk(text)
        else:
            raise ValueError(f"Unsupported chunking method: {self.method}")

    def _recursive_chunk(self, text: str) -> List[str]:
        """
        Uses LangChain's RecursiveCharacterTextSplitter to split text.

        :param text: str - The full document text.

        :return: List[str] - List of text chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],  # Prefer logical breaks like paragraphs or newlines
        )
        return text_splitter.split_text(text)

    def _sentence_chunk(self, text: str) -> List[str]:
        """
        Splits text into sentence-based chunks.

        :param text: str - The full document text.

        :return: List[str] - List of sentence-based text chunks.
        """
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If adding this sentence exceeds chunk size, save current chunk and start new one
            if current_length + sentence_length > self.chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

            current_chunk.append(sentence)
            current_length += sentence_length

        # Add any remaining chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks
