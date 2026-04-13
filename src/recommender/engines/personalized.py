"""Personalized recommendation engine."""

from typing import List, Dict, Any
from ..models import RecommendationRequest, RepositoryResult
from ..database import db_manager
from ..config import settings
from ..services.language_enricher import LanguageEnricherService
from .hybrid import HybridRetrievalEngine


class PersonalizedEngine(HybridRetrievalEngine):
    """
    Personalized engine that extends hybrid retrieval with user preferences.

    Personalization is applied as a re-ranking signal:
    - Boost repos in preferred languages
    - Boost repos with similar topics to past interactions
    - Only applies if user has minimum interactions
    - Never overrides hard constraints
    """

    def __init__(self, embedding_service=None, reranker_service=None):
        super().__init__(embedding_service, reranker_service)
        self.name = "personalized"
        self.version = "1.0.0"
        self.language_enricher = LanguageEnricherService()

    async def recommend(self, request: RecommendationRequest) -> List[RepositoryResult]:
        """Generate personalized recommendations."""

        # Enrich request with language preferences
        request = await self.language_enricher.enrich(request)

        # Get base recommendations from hybrid engine
        results = await super().recommend(request)

        # Apply personalization if enabled and user_id provided
        if request.enable_personalization and request.user_id and settings.enable_personalization:
            results = await self._apply_personalization(results, request)

        return results

    async def _apply_personalization(
        self, results: List[RepositoryResult], request: RecommendationRequest
    ) -> List[RepositoryResult]:
        """Apply personalization boost to results."""

        # Get user preferences
        prefs = await db_manager.get_user_preferences(request.user_id)

        if not prefs:
            return results

        # Check if user has enough interactions
        if prefs.total_interactions < settings.min_interactions_for_personalization:
            return results

        # Apply personalization boost
        for result in results:
            boost = 0.0

            # Language preference boost is now handled natively by hybrid.py via RRF

            # Topic preference boost (if available in repo data)
            # This would require topics to be part of the result

            # Apply boost to score
            original_score = result.score
            result.score = result.score * (1 + boost)

            # Update explanation
            if result.explanation:
                result.explanation["personalization_boost"] = boost
                result.explanation["original_score"] = original_score
                result.explanation["personalized"] = True

        # Re-sort by new scores
        results.sort(key=lambda x: x.score, reverse=True)

        # Update ranks
        for idx, result in enumerate(results):
            result.rank = idx + 1

        return results

    async def explain(self, repo_id: str, request: RecommendationRequest) -> Dict[str, Any]:
        """Explain personalized ranking."""
        explanation = await super().explain(repo_id, request)
        explanation["engine"] = self.name
        explanation["personalization"] = "User preferences applied to boost ranking"

        if request.user_id:
            # Fetch preferences once here; _apply_personalization fetches
            # independently during recommend() — both paths share no state.
            prefs = await db_manager.get_user_preferences(request.user_id)
            if prefs:
                explanation["user_interactions"] = prefs.total_interactions
                explanation["top_languages"] = dict(
                    sorted(
                        prefs.language_preferences.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:5]
                )

        return explanation
