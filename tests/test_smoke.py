"""Smoke tests — verify core modules import and instantiate without external services."""

import tempfile
from unittest.mock import patch


class TestMLOpsSmoke:
    """MLOps module smoke tests."""

    def test_mlflow_tracker_imports(self):
        from src.recommender.mlops.mlflow_tracker import MLflowTracker, create_tracker_from_env

        assert MLflowTracker is not None
        assert create_tracker_from_env is not None

    def test_mlflow_tracker_instantiates(self):
        """Tracker must instantiate without a running MLflow server (graceful degradation)."""
        with patch.dict("sys.modules", {"mlflow": None, "mlflow.tracking": None}):
            import importlib

            import src.recommender.mlops.mlflow_tracker as _mod

            importlib.reload(_mod)
            tracker = _mod.MLflowTracker(experiment_name="smoke-test", tracking_uri="./mlruns")
            assert tracker.experiment_name == "smoke-test"

    def test_drift_monitor_imports(self):
        from src.recommender.mlops.drift_monitor import DriftMonitor, create_monitor_from_env

        assert DriftMonitor is not None
        assert create_monitor_from_env is not None

    def test_drift_monitor_instantiates(self):
        from src.recommender.mlops.drift_monitor import DriftMonitor

        with tempfile.TemporaryDirectory() as tmp:
            monitor = DriftMonitor(report_dir=tmp)
            assert monitor.report_dir.exists()


class TestDataSmoke:
    """Data module smoke tests."""

    def test_feature_extractor_imports(self):
        from src.recommender.data.features import FeatureExtractor

        assert FeatureExtractor is not None

    def test_feature_extractor_instantiates(self):
        from src.recommender.data.features import FeatureExtractor

        fe = FeatureExtractor()
        assert fe is not None

    def test_repo_dataset_imports(self):
        from src.recommender.data.dataset import RepoDataset

        assert RepoDataset is not None

    def test_repo_dataset_from_empty_list(self):
        from src.recommender.data.dataset import RepoDataset

        ds = RepoDataset([])
        assert len(ds) == 0
