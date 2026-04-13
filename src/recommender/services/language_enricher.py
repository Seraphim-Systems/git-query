"""Service for enriching recommendation requests with user preferences."""

from typing import List
from ..models import RecommendationRequest
from ..database import db_manager
from ..config import settings


class LanguageEnricherService:
    """Enriches requests with language preferences to be used by retrieval engines."""

    async def enrich(self, request: RecommendationRequest) -> RecommendationRequest:
        """Add user preferred languages to the request."""

        # Initialize the list if it doesn't exist
        if not getattr(request, "preferred_languages", None):
            request.preferred_languages = []

        # Only enrich if we're doing personalization
        if not (request.enable_personalization and request.user_id and settings.enable_personalization):
            return request

        # Retrieve user preferences
        prefs = await db_manager.get_user_preferences(request.user_id)
        if not prefs:
            return request

        # Check if user has enough interactions or explicit preferences
        has_enough_interactions = prefs.total_interactions >= settings.min_interactions_for_personalization
        has_explicit = bool(prefs.explicit_languages)

        if not (has_enough_interactions or has_explicit):
            return request

        # Aggregate explicit languages and top learned languages
        explicit_set = set(prefs.explicit_languages or [])
        ranked = sorted(
            prefs.language_preferences.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )[:5]

        # Combine into a single list of preferred languages
        preferred = list(explicit_set.union({lang for lang, score in ranked}))
        request.preferred_languages = preferred

        return request
