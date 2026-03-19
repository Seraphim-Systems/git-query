"""Cross-encoder reranker service for accurate ranking of top candidates."""

import asyncio
import os
import logging
from typing import List, Optional
from ..config import settings
from ..models import RepositoryResult
from .adapters import BaseRerankerAdapter, AdapterFactory

logger = logging.getLogger(__name__)


class RerankerService:
    """
    Reranker service — delegates to a hot-swappable adapter backend.

    Supports CrossEncoder and LightGBM LambdaRank via the adapter pattern.
    The active backend is selected at load time by AdapterFactory.from_path().
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.cross_encoder_model_name
        self.model = None
        self.current_model_id: Optional[str] = None
        self._loaded_path: Optional[str] = None
        self._adapter: Optional[BaseRerankerAdapter] = None
        self._load_lock = asyncio.Lock()

    async def load_active_model(self, variant: str = "default"):
        """Load the currently active reranker model from the registry."""
        async with self._load_lock:
            from .registry_service import ModelRegistryService
            registry = ModelRegistryService()
            # Prefer explicitly-typed cross_encoder entries, but also allow generic 'reranker' entries
            active_model = await registry.get_active_model("cross_encoder", variant)
            if not active_model:
                active_model = await registry.get_active_model("reranker", variant)

            if not active_model:
                logger.warning(
                    "No active reranker model found for variant %r. Using default: %s",
                    variant,
                    self.model_name,
                )
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self.load_model, self.model_name)
                return

            if active_model.model_id == self.current_model_id:
                logger.info("Reranker model %s is already loaded.", active_model.model_id)
                return

            # Load from path (prefer model files stored under settings.model_path)
            full_path = os.path.join(settings.model_path, active_model.path) if getattr(active_model, 'path', None) else None
            loop = asyncio.get_running_loop()
            if full_path and os.path.exists(full_path):
                logger.info("Loading active reranker model: %s from %s", active_model.model_id, full_path)
                await loop.run_in_executor(None, self.load_model, full_path)
                self.current_model_id = active_model.model_id
            else:
                logger.warning(
                    "Active reranker model path not found on disk (model_id=%s, path=%s). Falling back to default: %s",
                    getattr(active_model, 'model_id', None),
                    getattr(active_model, 'path', None),
                    self.model_name,
                )
                await loop.run_in_executor(None, self.load_model, self.model_name)

    def load_model(self, model_path: str = None):
        """Load the reranker model into memory via AdapterFactory."""
        target = model_path or self.model_name
        logger.info("Loading reranker model: %s", target)
        self._adapter = AdapterFactory.from_path(target)
        # Keep self.model pointing at the adapter for legacy callers
        self.model = self._adapter
        self._loaded_path = target
        return self._adapter

    async def rerank(
        self,
        query: str,
        candidates: List[RepositoryResult],
        top_k: int = None,
    ) -> List[RepositoryResult]:
        """Rerank candidates using cross-encoder."""
        if not candidates:
            return []

        top_k = top_k or settings.rerank_top_k

        loop = asyncio.get_running_loop()
        scores = await loop.run_in_executor(None, self._adapter.score, query, candidates)

        for candidate, score in zip(candidates, scores):
            candidate.explanation = candidate.explanation or {}
            candidate.explanation["rerank_score"] = float(score)
            candidate.explanation["original_score"] = candidate.score
            candidate.score = float(score)

        reranked = sorted(candidates, key=lambda x: x.score, reverse=True)

        for idx, result in enumerate(reranked[:top_k]):
            result.rank = idx + 1
            result.explanation["reranked"] = True

        return reranked[:top_k]

