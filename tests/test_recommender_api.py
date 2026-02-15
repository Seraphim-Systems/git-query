"""Simplified API tests - focusing on contracts and functionality."""

import pytest
from datetime import datetime
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

from src.recommender.api import app
from src.recommender.models import UserInteraction, InteractionType


# ===== Test Client Setup =====

@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


# ===== Health & Basic Endpoints =====

class TestBasicEndpoints:
    """Test basic API endpoints."""

    def test_health_check(self, client):
        """Test health check endpoint returns correct structure."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["service"] == "recommender"
        assert data["version"] == "1.0.0"


# ===== Request Validation Tests =====

class TestRequestValidation:
    """Test that API validates requests correctly."""

    def test_recommendation_requires_query(self, client):
        """Test that query is required."""
        response = client.post("/recommend", json={})
        assert response.status_code == 422  # Validation error

    def test_recommendation_validates_top_k(self, client):
        """Test that top_k is validated."""
        request_data = {
            "query": "test",
            "top_k": 0,  # Invalid: too small
        }
        response = client.post("/recommend", json=request_data)
        assert response.status_code == 422

        request_data = {
            "query": "test",
            "top_k": 100,  # Invalid: too large
        }
        response = client.post("/recommend", json=request_data)
        assert response.status_code == 422

    def test_interaction_requires_fields(self, client):
        """Test that interaction logging validates required fields."""
        response = client.post("/interaction", json={})
        assert response.status_code == 422

    def test_interaction_validates_type(self, client):
        """Test that interaction type is validated."""
        interaction_data = {
            "user_id": "user123",
            "query": "python web",
            "repo_id": "repo1",
            "interaction_type": "invalid_type",  # Invalid
            "variant": "baseline",
        }
        response = client.post("/interaction", json=interaction_data)
        assert response.status_code == 422


# ===== Response Structure Tests =====

class TestResponseStructure:
    """Test that API responses have correct structure (without needing real data)."""

    def test_interaction_response_structure(self, client):
        """Test interaction endpoint returns correct structure."""
        with patch('src.recommender.api.db_manager') as mock_db:
            mock_db.log_interaction = AsyncMock(return_value="interaction123")

            interaction_data = {
                "user_id": "user123",
                "query": "python web",
                "repo_id": "repo1",
                "interaction_type": "click",
                "position_in_results": 1,
                "variant": "baseline",
            }

            response = client.post("/interaction", json=interaction_data)
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
            assert "interaction_id" in data
            assert data["status"] == "success"

    def test_metrics_not_found(self, client):
        """Test metrics endpoint returns 404 when no metrics exist."""
        with patch('src.recommender.api.db_manager') as mock_db:
            mock_db.get_latest_metrics = AsyncMock(return_value=None)

            response = client.get("/metrics/baseline")
            assert response.status_code == 404

    def test_preferences_not_found(self, client):
        """Test preferences endpoint returns 404 when user not found."""
        with patch('src.recommender.api.db_manager') as mock_db:
            mock_db.get_user_preferences = AsyncMock(return_value=None)

            response = client.get("/preferences/nonexistent_user")
            assert response.status_code == 404


# ===== Import Tests =====

class TestImports:
    """Test that all modules can be imported without errors."""

    def test_import_api(self):
        """Test that API module imports successfully."""
        from src.recommender import api
        assert api.app is not None

    def test_import_models(self):
        """Test that models module imports successfully."""
        from src.recommender import models
        assert models.RecommendationRequest is not None

    def test_import_engines(self):
        """Test that engines module imports successfully."""
        from src.recommender import engines
        assert engines.BaselineEngine is not None

    def test_import_services(self):
        """Test that services module imports successfully."""
        from src.recommender import services
        assert services.EmbeddingService is not None

    def test_import_database(self):
        """Test that database module imports successfully."""
        from src.recommender import database
        assert database.db_manager is not None
