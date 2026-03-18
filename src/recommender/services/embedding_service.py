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
        self._loaded_path: Optional[str] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_lock = asyncio.Lock()

    async def load_active_model(self, variant: str = "default"):
        """Load the currently active embedding model from the registry."""
        async with self._load_lock:
            from .registry_service import ModelRegistryService
            registry = ModelRegistryService()

            active_model = await registry.get_active_model("embedding", variant)

            if not active_model:
                logger.warning(
                    "No active embedding model found for variant %r. Using default: %s",
                    variant,
                    self.model_name,
                )
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self.load_model, self.model_name)
                return

            if active_model.model_id == self.current_model_id:
                logger.info("Embedding model %s is already loaded.", active_model.model_id)
                return

            full_path = os.path.join(settings.model_path, active_model.path)
            loop = asyncio.get_running_loop()
            if os.path.exists(full_path):
                logger.info(
                    "Loading active embedding model: %s from %s",
                    active_model.model_id,
                    full_path,
                )
                await loop.run_in_executor(None, self.load_model, full_path)
                self.current_model_id = active_model.model_id
            else:
                logger.error(
                    "Active model path not found: %s. Falling back to default.", full_path
                )
                await loop.run_in_executor(None, self.load_model, self.model_name)

    def load_model(self, model_path: str = None):
        """Load the embedding model into memory."""
        target = model_path or self.model_name
        if self.model is None or target != self._loaded_path:
            logger.info("Initializing SentenceTransformer with: %s", target)
            self.model = SentenceTransformer(target, device=self.device)
            self._loaded_path = target
        return self.model

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(None, self._embed_sync, text)
        return embedding.tolist()

    def _embed_sync(self, text: str):
        """Synchronous embedding generation."""
        model = self.load_model()
        return model.encode(text, convert_to_tensor=False, show_progress_bar=False)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        loop = asyncio.get_running_loop()
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
