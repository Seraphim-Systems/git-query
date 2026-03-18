"""Smoke tests — verify MLOps modules import and instantiate without external services.

Uses direct file loading (importlib) to bypass src/recommender/__init__.py,
which imports engines and config that require torch, pydantic-settings, etc.
The MLOps modules themselves have no such heavy dependencies.
"""

import importlib.util
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).parent.parent


def _load(module_name: str, rel_path: str):
    """Load a Python file directly, bypassing package __init__ imports."""
    file_path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class TestMLOpsSmoke:
    """MLOps module smoke tests — no torch or pydantic-settings required."""

    def test_mlflow_tracker_imports(self):
        with patch.dict("sys.modules", {"mlflow": None, "mlflow.tracking": None}):
            mod = _load("mlflow_tracker_smoke", "src/recommender/mlops/mlflow_tracker.py")
            assert hasattr(mod, "MLflowTracker")
            assert hasattr(mod, "create_tracker_from_env")

    def test_mlflow_tracker_instantiates(self):
        with patch.dict("sys.modules", {"mlflow": None, "mlflow.tracking": None}):
            mod = _load("mlflow_tracker_smoke2", "src/recommender/mlops/mlflow_tracker.py")
            tracker = mod.MLflowTracker(experiment_name="smoke-test", tracking_uri="./mlruns")
            assert tracker.experiment_name == "smoke-test"

    def test_drift_monitor_imports(self):
        mod = _load("drift_monitor_smoke", "src/recommender/mlops/drift_monitor.py")
        assert hasattr(mod, "DriftMonitor")
        assert hasattr(mod, "create_monitor_from_env")

    def test_drift_monitor_instantiates(self):
        mod = _load("drift_monitor_smoke2", "src/recommender/mlops/drift_monitor.py")
        with tempfile.TemporaryDirectory() as tmp:
            monitor = mod.DriftMonitor(report_dir=tmp)
            assert monitor.report_dir.exists()


class TestDataSmoke:
    """Data module smoke tests — stdlib only, no heavy deps."""

    def test_feature_extractor_imports(self):
        mod = _load("features_smoke", "src/recommender/data/features.py")
        assert hasattr(mod, "FeatureExtractor")

    def test_feature_extractor_instantiates(self):
        mod = _load("features_smoke2", "src/recommender/data/features.py")
        fe = mod.FeatureExtractor()
        assert fe is not None

    def test_repo_dataset_imports(self):
        mod = _load("dataset_smoke", "src/recommender/data/dataset.py")
        assert hasattr(mod, "RepoDataset")

    def test_repo_dataset_from_empty_list(self):
        mod = _load("dataset_smoke2", "src/recommender/data/dataset.py")
        ds = mod.RepoDataset([])
        assert len(ds) == 0
