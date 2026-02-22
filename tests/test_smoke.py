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


def test_mlops_imports():
    """
    Verify that MLOps modules can be imported.
    """
    try:
        from src.recommender.mlops import MLflowTracker, DriftMonitor
    except ImportError as e:
        pytest.fail(f"Failed to import MLOps components: {e}")


def test_mlflow_tracker_creation():
    """
    Verify that MLflowTracker can be instantiated.
    """
    try:
        from src.recommender.mlops.mlflow_tracker import MLflowTracker
        tracker = MLflowTracker(experiment_name="smoke-test")
        assert tracker is not None
        assert tracker.experiment_name == "smoke-test"
    except Exception as e:
        pytest.fail(f"Failed to create MLflowTracker: {e}")


def test_drift_monitor_creation():
    """
    Verify that DriftMonitor can be instantiated.
    """
    import tempfile
    try:
        from src.recommender.mlops.drift_monitor import DriftMonitor
        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = DriftMonitor(report_dir=tmpdir)
            assert monitor is not None
    except Exception as e:
        pytest.fail(f"Failed to create DriftMonitor: {e}")
