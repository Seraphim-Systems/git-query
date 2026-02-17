"""Embedding service for semantic search."""

import os
from typing import List, Optional
import torch
from sentence_transformers import SentenceTransformer
from ..config import settings
from ..models import ModelMetadata
import asyncio
import logging

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating embeddings using sentence transformers."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.embedding_model_name
        self.model = None
        self.current_model_id: Optional[str] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    async def load_active_model(self, variant: str = "default"):
        """Load the currently active embedding model from the registry."""
        from .registry_service import ModelRegistryService
        registry = ModelRegistryService()
        
        active_model = await registry.get_active_model("embedding", variant)
        
        if not active_model:
            logger.warning(f"No active embedding model found for variant '{variant}'. Using default: {self.model_name}")
            self.load_model(self.model_name)
            return

        if active_model.model_id == self.current_model_id:
            logger.info(f"Embedding model {active_model.model_id} is already loaded.")
            return

        # Load from path
        full_path = os.path.join(settings.model_path, active_model.path)
        if os.path.exists(full_path):
            logger.info(f"Loading active embedding model: {active_model.model_id} from {full_path}")
            self.load_model(full_path)
            self.current_model_id = active_model.model_id
        else:
            logger.error(f"Active model path not found: {full_path}. Falling back to default.")
            self.load_model(self.model_name)

    def load_model(self, model_path: str = None):
        """Load the embedding model into memory."""
        target = model_path or self.model_name
        if self.model is None or target != self.model_name:
            logger.info(f"Initializing SentenceTransformer with: {target}")
            self.model = SentenceTransformer(target, device=self.device)
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

