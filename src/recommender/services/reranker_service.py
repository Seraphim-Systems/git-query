"""Cross-encoder reranker service for accurate ranking of top candidates."""

from typing import List, Tuple
from sentence_transformers import CrossEncoder
from ..config import settings
from ..models import RepositoryResult
import asyncio


class RerankerService:
    """
    Cross-encoder based reranker.

    More accurate but slower than bi-encoders.
    Used to rerank top K candidates from retrieval.
    """

    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.cross_encoder_model_name
        self.model = None

    def load_model(self):
        """Load the cross-encoder model."""
        if self.model is None:
            self.model = CrossEncoder(self.model_name)
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

        # Run reranking in thread pool
        loop = asyncio.get_event_loop()
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
        parts = [repo.name]

        if repo.description:
            parts.append(repo.description)

        if repo.language:
            parts.append(f"Language: {repo.language}")

        parts.append(f"Stars: {repo.stars}")

        return " | ".join(parts)

