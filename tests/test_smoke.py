import pytest

def test_recommender_imports():
    """
    Verify that the core recommender components can be imported.
    This ensures that the PYTHONPATH is correct and dependencies are installed.
    """
    try:
        from src.recommender import (
            RecommendationEngine,
            BaselineEngine,
            HybridRetrievalEngine,
            PersonalizedEngine,
            settings
        )
    except ImportError as e:
        pytest.fail(f"Failed to import recommender components: {e}")

def test_database_imports():
    """
    Verify that database-related modules can be imported.
    """
    try:
        from src.recommender.database import DatabaseManager
    except ImportError as e:
        pytest.fail(f"Failed to import DatabaseManager: {e}")

def test_services_imports():
    """
    Verify that service modules can be imported.
    """
    try:
        from src.recommender.services import (
            ab_test_service,
            embedding_service,
            personalization_service,
            reranker_service
        )
    except ImportError as e:
        pytest.fail(f"Failed to import services: {e}")
