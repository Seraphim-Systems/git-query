"""Embedding service for semantic search."""

from typing import List
import torch
from sentence_transformers import SentenceTransformer
from ..config import settings
import asyncio
from functools import lru_cache


class EmbeddingService:
    """Service for generating embeddings using sentence transformers."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.embedding_model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        """Load the embedding model."""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name, device=self.device)
        return self.model

    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding
        """
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, self._embed_sync, text)
        return embedding.tolist()

    def _embed_sync(self, text: str):
        """Synchronous embedding generation."""
        model = self.load_model()
        return model.encode(text, convert_to_tensor=False, show_progress_bar=False)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings
        """
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, self._embed_batch_sync, texts)
        return embeddings.tolist()

    def _embed_batch_sync(self, texts: List[str]):
        """Synchronous batch embedding generation."""
        model = self.load_model()
        return model.encode(
            texts,
            batch_size=settings.batch_size,
            convert_to_tensor=False,
            show_progress_bar=False,
        )

    def get_dimension(self) -> int:
        """Get embedding dimension."""
        model = self.load_model()
        return model.get_sentence_embedding_dimension()

