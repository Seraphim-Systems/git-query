"""Simplified functional tests for the recommendation system."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from src.recommender.models import (
    RecommendationRequest,
    RepositoryResult,
    UserInteraction,
    InteractionType,
    UserPreferences,
)


# ===== Model Validation Tests =====

class TestModels:
    """Test that Pydantic models validate correctly."""

    def test_recommendation_request_validation(self):
        """Test RecommendationRequest validates correctly."""
        request = RecommendationRequest(
            query="python web framework",
            top_k=10,
        )
        assert request.query == "python web framework"
        assert request.top_k == 10
        assert request.enable_personalization is True

    def test_recommendation_request_top_k_limits(self):
        """Test RecommendationRequest enforces top_k limits."""
        with pytest.raises(Exception):
            RecommendationRequest(query="test", top_k=0)

        with pytest.raises(Exception):
            RecommendationRequest(query="test", top_k=100)

    def test_repository_result_creation(self):
        """Test RepositoryResult can be created."""
        result = RepositoryResult(
            repo_id="repo1",
            name="django",
            full_name="django/django",
            description="Web framework",
            language="Python",
            stars=75000,
            forks=28000,
            url="https://github.com/django/django",
            license="BSD-3-Clause",
            last_updated=datetime(2026, 2, 10),
            score=0.95,
            rank=1,
        )
        assert result.repo_id == "repo1"
        assert result.score == 0.95
        assert result.rank == 1

    def test_user_interaction_types(self):
        """Test all interaction types are valid."""
        interaction = UserInteraction(
            user_id="user123",
            query="test",
            repo_id="repo1",
            interaction_type=InteractionType.CLICK,
            position_in_results=1,
            variant="baseline",
        )
        assert interaction.interaction_type == InteractionType.CLICK

        for itype in [InteractionType.SAVE, InteractionType.THUMBS_UP,
                      InteractionType.THUMBS_DOWN, InteractionType.DISMISS]:
            interaction.interaction_type = itype
            assert interaction.interaction_type == itype

    def test_user_preferences_model(self):
        """Test UserPreferences model."""
        prefs = UserPreferences(
            user_id="user123",
            language_preferences={"Python": 0.8, "JavaScript": 0.2},
            topic_preferences={"web": 0.9, "api": 0.7},
            total_interactions=25,
        )
        assert prefs.user_id == "user123"
        assert prefs.total_interactions == 25
        assert prefs.language_preferences["Python"] == 0.8


# ===== Basic Engine Tests (Structure Only) =====

class TestEngineStructure:
    """Test that engines have the correct structure and can be instantiated."""

    def test_baseline_engine_exists(self):
        """Test baseline engine can be imported and instantiated."""
        from src.recommender.engines.baseline import BaselineEngine
        engine = BaselineEngine()
        assert engine is not None
        assert engine.name == "baseline"

    def test_hybrid_engine_exists(self):
        """Test hybrid engine can be imported and instantiated."""
        from src.recommender.engines.hybrid import HybridRetrievalEngine
        engine = HybridRetrievalEngine()
        assert engine is not None
        assert engine.name == "hybrid"

    def test_personalized_engine_exists(self):
        """Test personalized engine can be imported and instantiated."""
        from src.recommender.engines.personalized import PersonalizedEngine
        engine = PersonalizedEngine()
        assert engine is not None
        assert engine.name == "personalized"


# ===== Service Structure Tests =====

class TestServices:
    """Test that services can be instantiated."""

    def test_embedding_service_exists(self):
        """Test embedding service can be imported."""
        from src.recommender.services import EmbeddingService
        service = EmbeddingService()
        assert service is not None

    def test_reranker_service_exists(self):
        """Test reranker service can be imported."""
        from src.recommender.services import RerankerService
        service = RerankerService()
        assert service is not None

    def test_personalization_service_exists(self):
        """Test personalization service can be imported."""
        from src.recommender.services import PersonalizationService
        service = PersonalizationService()
        assert service is not None

    def test_ab_test_service_exists(self):
        """Test A/B test service can be imported."""
        from src.recommender.services import ABTestService
        service = ABTestService()
        assert service is not None
