"""Cross-encoder reranker service for accurate ranking of top candidates."""

import os
import logging
from typing import List, Optional
from sentence_transformers import CrossEncoder
from ..config import settings
from ..models import RepositoryResult, ModelMetadata
import asyncio

logger = logging.getLogger(__name__)


class RerankerService:
    """
    Cross-encoder based reranker.

    More accurate but slower than bi-encoders.
    Used to rerank top K candidates from retrieval.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.cross_encoder_model_name
        self.model = None
        self.current_model_id: Optional[str] = None
        self._loaded_path: Optional[str] = None

    async def load_active_model(self, variant: str = "default"):
        """Load the currently active reranker model from the registry."""
        from .registry_service import ModelRegistryService
        registry = ModelRegistryService()

        active_model = await registry.get_active_model("cross_encoder", variant)

        if not active_model:
            logger.warning(
                "No active reranker model found for variant %r. Using default: %s",
                variant,
                self.model_name,
            )
            self.load_model(self.model_name)
            return

        if active_model.model_id == self.current_model_id:
            logger.info("Reranker model %s is already loaded.", active_model.model_id)
            return

        full_path = os.path.join(settings.model_path, active_model.path)
        if os.path.exists(full_path):
            logger.info(
                "Loading active reranker model: %s from %s",
                active_model.model_id,
                full_path,
            )
            self.load_model(full_path)
            self.current_model_id = active_model.model_id
        else:
            logger.error(
                "Active reranker model path not found: %s. Falling back to default.",
                full_path,
            )
            self.load_model(self.model_name)

    def load_model(self, model_path: str = None):
        """Load the cross-encoder model into memory."""
        target = model_path or self.model_name
        if self.model is None or target != self._loaded_path:
            logger.info("Initializing CrossEncoder with: %s", target)
            self.model = CrossEncoder(target)
            self._loaded_path = target
        return self.model

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

        pairs = [
            [query, self._create_repo_text(candidate)]
            for candidate in candidates
        ]

        loop = asyncio.get_running_loop()
        scores = await loop.run_in_executor(None, self._score_pairs, pairs)

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

    def _score_pairs(self, pairs: List[List[str]]) -> List[float]:
        """Score query-document pairs using cross-encoder."""
        model = self.load_model()
        scores = model.predict(pairs, show_progress_bar=False)
        return scores

    def _create_repo_text(self, repo: RepositoryResult) -> str:
        """Create text representation of repository for reranking."""
        from ..training.utils import prepare_repo_text
        repo_dict = {
            "name": repo.name,
            "description": repo.description,
            "language": repo.language,
        }
        return prepare_repo_text(repo_dict)
