"""Cross-encoder reranker service for accurate ranking of top candidates."""

import os
import logging
from typing import List, Tuple, Optional
from sentence_transformers import CrossEncoder
from ..config import settings
from ..models import RepositoryResult, ModelMetadata
import asyncio
import joblib
import pandas as pd
from ..data.features import FeatureExtractor

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

    async def load_active_model(self, variant: str = "default"):
        """Load the currently active reranker model from the registry."""
        from .registry_service import ModelRegistryService
        registry = ModelRegistryService()
        # Prefer explicitly-typed cross_encoder entries, but also allow generic 'reranker' entries
        active_model = await registry.get_active_model("cross_encoder", variant)
        if not active_model:
            active_model = await registry.get_active_model("reranker", variant)

        if not active_model:
            logger.warning(f"No active reranker model found for variant '{variant}'. Using default: {self.model_name}")
            self.load_model(self.model_name)
            return

        if active_model.model_id == self.current_model_id:
            logger.info(f"Reranker model {active_model.model_id} is already loaded.")
            return

        # Load from path (prefer model files stored under settings.model_path)
        full_path = os.path.join(settings.model_path, active_model.path) if getattr(active_model, 'path', None) else None
        if full_path and os.path.exists(full_path):
            logger.info(f"Loading active reranker model: {active_model.model_id} from {full_path}")
            self.load_model(full_path)
            self.current_model_id = active_model.model_id
        else:
            # If registry holds a model identifier (eg. HF id or local dir), try to load it directly
            try:
                logger.info(f"Attempting to load active reranker by id/path: {getattr(active_model, 'path', None)}")
                self.load_model(getattr(active_model, 'path', self.model_name))
                self.current_model_id = active_model.model_id
            except Exception:
                logger.error(f"Could not load active reranker model '{getattr(active_model,'model_id', None)}'. Falling back to default.")
                self.load_model(self.model_name)

    def load_model(self, model_path: str = None):
        """Load the cross-encoder model into memory."""
        target = model_path or self.model_name
        # If target is a LightGBM artifact (pickled), load and wrap it with a small adapter
        if isinstance(target, str) and (target.endswith('.pkl') or target.endswith('.joblib')):
            logger.info(f"Loading LightGBM reranker artifact via joblib: {target}")
            model_obj = joblib.load(target)

            # Adapter exposes a candidate-aware prediction method used by rerank()
            class _LightGBMAdapter:
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
            return self.model

        # Otherwise treat as a CrossEncoder model id/path
        if self.model is None or target != self.model_name:
            logger.info(f"Initializing CrossEncoder with: {target}")
            self.model = CrossEncoder(target)
        return self.model

    async def rerank(
        self,
        query: str,
        candidates: List[RepositoryResult],
        top_k: int = None,
    ) -> List[RepositoryResult]:
        """
        Rerank candidates using cross-encoder.

        Args:
            query: User query
            candidates: List of candidate repositories
            top_k: Number of top results to return

        Returns:
            Reranked list of repositories
        """
        if not candidates:
            return []

        top_k = top_k or settings.rerank_top_k

        # Prepare query-document pairs
        pairs = []
        for candidate in candidates:
            # Create a text representation of the repo
            repo_text = self._create_repo_text(candidate)
            pairs.append([query, repo_text])

        # Run reranking in thread pool. If a LightGBM adapter is loaded, call its
        # candidate-aware prediction method so features are computed from RepositoryResult.
        loop = asyncio.get_event_loop()
        if hasattr(self.model, 'predict_from_candidates'):
            scores = await loop.run_in_executor(None, self.model.predict_from_candidates, query, candidates)
        else:
            scores = await loop.run_in_executor(None, self._score_pairs, pairs)

        # Attach scores and sort
        for candidate, score in zip(candidates, scores):
            candidate.explanation = candidate.explanation or {}
            candidate.explanation["rerank_score"] = float(score)
            candidate.explanation["original_score"] = candidate.score
            candidate.score = float(score)

        # Sort by new score
        reranked = sorted(candidates, key=lambda x: x.score, reverse=True)

        # Update ranks
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

