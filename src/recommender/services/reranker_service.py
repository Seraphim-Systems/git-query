"""Cross-encoder reranker service for accurate ranking of top candidates."""

import asyncio
import os
import logging
from typing import List, Optional
from sentence_transformers import CrossEncoder
from ..config import settings
from ..models import RepositoryResult, ModelMetadata
import joblib
import pandas as pd
from ..data.features import FeatureExtractor
from ..training.utils import prepare_repo_text

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
        self._is_lgbm: bool = False  # set to True when a .pkl LightGBM adapter is loaded
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
        """Load the cross-encoder model into memory."""
        target = model_path or self.model_name
        # If target is a LightGBM artifact (pickled), load and wrap it with a small adapter
        if isinstance(target, str) and (target.endswith('.pkl') or target.endswith('.joblib')):
            logger.info("Loading LightGBM reranker artifact via joblib: %s", target)
            model_obj = joblib.load(target)

            # Adapter exposes a candidate-aware prediction method used by rerank()
            class _LightGBMAdapter:
                _is_lgbm_adapter = True  # sentinel — avoids hasattr() collision with MagicMock

                def __init__(self, model, feature_extractor: FeatureExtractor):
                    self.model = model
                    self.fe = feature_extractor

                def predict_from_candidates(self, query: str, candidates: List[RepositoryResult]):
                    # Build a DataFrame from RepositoryResult objects with columns expected by FeatureExtractor
                    rows = []
                    for c in candidates:
                        rows.append({
                            'name': getattr(c, 'name', None),
                            'description': getattr(c, 'description', None),
                            'stars': getattr(c, 'stars', None),
                            'forks': getattr(c, 'forks', None),
                            'language': getattr(c, 'language', None),
                            'license': getattr(c, 'license', None),
                            'topics': getattr(c, 'topics', None),
                            'readme': getattr(c, 'readme', None),
                            'updated_at': getattr(c, 'updated_at', None),
                            'pushed_at': getattr(c, 'pushed_at', None),
                        })
                    df = pd.DataFrame(rows)
                    X = self.fe.extract_all(df, query=query)
                    # If the saved model has an explicit feature ordering, respect it
                    if hasattr(self.model, 'feature_cols'):
                        try:
                            X = X[self.model.feature_cols]
                        except Exception:
                            pass
                    return self.model.predict(X.values)

            self.model = _LightGBMAdapter(model_obj, FeatureExtractor())
            self._is_lgbm = True
            self._loaded_path = target
            return self.model

        # Otherwise treat as a CrossEncoder model id/path
        if self.model is None or target != self.model_name:
            logger.info("Initializing CrossEncoder with: %s", target)
            self.model = CrossEncoder(target)
            self._loaded_path = target
        self._is_lgbm = False
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

        # Run reranking in thread pool. If a LightGBM adapter is loaded, call its
        # candidate-aware prediction method so features are computed from RepositoryResult.
        loop = asyncio.get_running_loop()
        if self._is_lgbm:
            scores = await loop.run_in_executor(None, self.model.predict_from_candidates, query, candidates)
        else:
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
        scores = self.model.predict(pairs, show_progress_bar=False)
        return scores

    def _create_repo_text(self, repo: RepositoryResult) -> str:
        """Create text representation of repository for reranking."""
        repo_dict = {
            "name": repo.name,
            "description": repo.description,
            "language": repo.language,
        }
        return prepare_repo_text(repo_dict)
