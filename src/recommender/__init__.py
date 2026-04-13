"""Git-Query Recommendation System.

AI-powered repository recommendation with hybrid retrieval,
personalization, and A/B testing.
"""

from __future__ import annotations

from typing import Any

__version__ = "1.0.0"

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


def __getattr__(name: str) -> Any:
    if name == "settings":
        from .config import settings

        return settings
    if name in {
        "RecommendationRequest",
        "RecommendationResponse",
        "UserInteraction",
        "InteractionType",
    }:
        from .models import (
            InteractionType,
            RecommendationRequest,
            RecommendationResponse,
            UserInteraction,
        )

        return {
            "RecommendationRequest": RecommendationRequest,
            "RecommendationResponse": RecommendationResponse,
            "UserInteraction": UserInteraction,
            "InteractionType": InteractionType,
        }[name]
    if name in {
        "RecommendationEngine",
        "BaselineEngine",
        "HybridRetrievalEngine",
        "PersonalizedEngine",
    }:
        from .engines import (
            BaselineEngine,
            HybridRetrievalEngine,
            PersonalizedEngine,
            RecommendationEngine,
        )

        return {
            "RecommendationEngine": RecommendationEngine,
            "BaselineEngine": BaselineEngine,
            "HybridRetrievalEngine": HybridRetrievalEngine,
            "PersonalizedEngine": PersonalizedEngine,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
