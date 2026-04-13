"""Git-Query Recommendation System.

AI-powered repository recommendation with hybrid retrieval,
personalization, and A/B testing.
"""

__version__ = "1.0.0"

from .config import settings
from .models import (
    RecommendationRequest,
    RecommendationResponse,
    UserInteraction,
    InteractionType,
)
from .engines import (
    RecommendationEngine,
    BaselineEngine,
    HybridRetrievalEngine,
    PersonalizedEngine,
)

__all__ = [
    "settings",
    "RecommendationRequest",
    "RecommendationResponse",
    "UserInteraction",
    "InteractionType",
    "RecommendationEngine",
    "BaselineEngine",
    "HybridRetrievalEngine",
    "PersonalizedEngine",
]
