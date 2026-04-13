"""Tests for MLflow tracking integration."""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestMLflowTracker:
    """Test suite for MLflowTracker."""

    def test_import_mlflow_tracker(self):
        """Test that MLflowTracker can be imported."""
        from src.recommender.mlops.mlflow_tracker import MLflowTracker

        assert MLflowTracker is not None

    def test_tracker_initialization_without_mlflow(self):
        """Test tracker initializes gracefully when MLflow is not available."""
        with patch.dict("sys.modules", {"mlflow": None}):
            # Re-import to test without mlflow
            from src.recommender.mlops import mlflow_tracker

            # Should not raise
            tracker = mlflow_tracker.MLflowTracker(experiment_name="test-experiment", tracking_uri="./test_mlruns")
            assert tracker.experiment_name == "test-experiment"

    @pytest.mark.skipif(not os.getenv("MLFLOW_AVAILABLE", "false").lower() == "true", reason="MLflow not available")
    def test_tracker_with_mlflow(self):
        """Test tracker with actual MLflow (when available)."""
        from src.recommender.mlops.mlflow_tracker import MLFLOW_AVAILABLE, MLflowTracker

        if not MLFLOW_AVAILABLE:
            pytest.skip("MLflow not installed")

        tracker = MLflowTracker(experiment_name="test-experiment", tracking_uri="./test_mlruns")

        # Test logging params without active run (should be no-op)
        tracker.log_params({"test_param": "value"})

    def test_log_params_converts_to_string(self):
        """Test that params are converted to strings."""
        from src.recommender.mlops.mlflow_tracker import MLflowTracker

        tracker = MLflowTracker(experiment_name="test")

        # Should not raise even with non-string values
        tracker.log_params(
            {
                "int_param": 42,
                "float_param": 3.14,
                "bool_param": True,
                "list_param": [1, 2, 3],
            }
        )

    def test_log_metrics_handles_no_run(self):
        """Test that logging metrics without active run doesn't raise."""
        from src.recommender.mlops.mlflow_tracker import MLflowTracker

        tracker = MLflowTracker(experiment_name="test")

        # Should not raise
        tracker.log_metrics({"precision": 0.85, "recall": 0.72})

    def test_log_model_info_separates_params_and_metrics(self):
        """Test that model info is correctly categorized."""
        from src.recommender.mlops.mlflow_tracker import MLflowTracker

        tracker = MLflowTracker(experiment_name="test")

        model_metadata = {
            "model_name": "all-MiniLM-L6-v2",
            "embedding_dim": 384,
            "num_repos": 1000,
            "training_time_seconds": 45.5,
            "device": "cuda",
            "normalized": True,
            "batch_size": 32,
        }

        # Should not raise
        tracker.log_model_info(model_metadata)

    def test_log_evaluation_metrics_flattens_dict(self):
        """Test that nested evaluation metrics are flattened."""
        from src.recommender.mlops.mlflow_tracker import MLflowTracker

        tracker = MLflowTracker(experiment_name="test")

        eval_metrics = {
            "precision_at_k": {1: 0.6, 5: 0.75, 10: 0.82},
            "recall_at_k": {1: 0.1, 5: 0.35, 10: 0.55},
            "ndcg_at_k": {1: 0.6, 5: 0.7, 10: 0.78},
            "mrr": 0.65,
        }

        # Should not raise
        tracker.log_evaluation_metrics(eval_metrics)

    def test_create_tracker_from_env(self):
        """Test creating tracker from environment variables."""
        from src.recommender.mlops.mlflow_tracker import create_tracker_from_env

        with patch.dict(
            os.environ,
            {
                "MLFLOW_EXPERIMENT_NAME": "env-test-experiment",
                "MLFLOW_TRACKING_URI": "./env_mlruns",
            },
        ):
            tracker = create_tracker_from_env()
            assert tracker.experiment_name == "env-test-experiment"
            assert tracker.tracking_uri == "./env_mlruns"

    def test_get_run_id_returns_none_without_run(self):
        """Test that get_run_id returns None when no run is active."""
        from src.recommender.mlops.mlflow_tracker import MLflowTracker

        tracker = MLflowTracker(experiment_name="test")
        assert tracker.get_run_id() is None


class TestLogArtifact:
    def test_no_op_without_active_run(self, tmp_path):
        """log_artifact is a silent no-op when no MLflow run is active."""
        from src.recommender.mlops.mlflow_tracker import MLflowTracker

        tracker = MLflowTracker(experiment_name="test")
        f = tmp_path / "artifact.txt"
        f.write_text("data")

        # Should not raise
        tracker.log_artifact(str(f))

    def test_calls_log_artifact_for_file(self, tmp_path):
        from src.recommender.mlops.mlflow_tracker import MLFLOW_AVAILABLE, MLflowTracker

        if not MLFLOW_AVAILABLE:
            pytest.skip("MLflow not installed")

        tracker = MLflowTracker(experiment_name="test")
        tracker._run = MagicMock()  # simulate active run
        f = tmp_path / "model.pkl"
        f.write_text("model")

        with patch("mlflow.log_artifact") as mock_log:
            tracker.log_artifact(str(f))

        mock_log.assert_called_once_with(str(f), None)

    def test_calls_log_artifacts_for_directory(self, tmp_path):
        from src.recommender.mlops.mlflow_tracker import MLFLOW_AVAILABLE, MLflowTracker

        if not MLFLOW_AVAILABLE:
            pytest.skip("MLflow not installed")

        tracker = MLflowTracker(experiment_name="test")
        tracker._run = MagicMock()  # simulate active run

        with patch("mlflow.log_artifacts") as mock_log:
            tracker.log_artifact(str(tmp_path), artifact_path="models/")

        mock_log.assert_called_once_with(str(tmp_path), "models/")

    def test_swallows_mlflow_exception(self, tmp_path):
        """Errors from MLflow are caught and don't propagate."""
        from src.recommender.mlops.mlflow_tracker import MLFLOW_AVAILABLE, MLflowTracker

        if not MLFLOW_AVAILABLE:
            pytest.skip("MLflow not installed")

        tracker = MLflowTracker(experiment_name="test")
        tracker._run = MagicMock()
        f = tmp_path / "artifact.txt"
        f.write_text("data")

        with patch("mlflow.log_artifact", side_effect=RuntimeError("mlflow error")):
            tracker.log_artifact(str(f))  # must not raise


class TestMLflowTrackerContextManager:
    """Test the context manager functionality."""

    def test_start_run_context_manager(self):
        """Test using start_run as context manager."""
        from src.recommender.mlops.mlflow_tracker import MLFLOW_AVAILABLE, MLflowTracker

        tracker = MLflowTracker(experiment_name="test")

        with tracker.start_run(run_name="test-run") as run:
            # When MLflow is not available, run should be None
            if not MLFLOW_AVAILABLE:
                assert run is None

    def test_nested_runs(self):
        """Test nested run capability."""
        from src.recommender.mlops.mlflow_tracker import MLFLOW_AVAILABLE, MLflowTracker

        if not MLFLOW_AVAILABLE:
            pytest.skip("MLflow not installed")

        tracker = MLflowTracker(experiment_name="test")

        with tracker.start_run(run_name="parent"):
            with tracker.start_run(run_name="child", nested=True):
                pass  # Should not raise


# ---------------------------------------------------------------------------
# P4: register_model_version — full path with real MLflow file-store
# ---------------------------------------------------------------------------


class TestRegisterModelVersion:
    def test_returns_none_when_mlflow_unavailable(self):
        from src.recommender.mlops.mlflow_tracker import MLflowTracker

        tracker = MLflowTracker(experiment_name="test")
        tracker._client = None  # simulate unavailable

        result = tracker.register_model_version(
            run_id="fake-run-id",
            model_name="test-model",
            artifact_path="models/lgbm.pkl",
        )

        assert result is None

    def test_returns_version_number_on_success(self, tmp_path):
        from src.recommender.mlops.mlflow_tracker import MLFLOW_AVAILABLE, MLflowTracker

        if not MLFLOW_AVAILABLE:
            pytest.skip("MLflow not installed")

        mock_client = MagicMock()
        mock_mv = MagicMock()
        mock_mv.version = "3"
        mock_client.create_model_version.return_value = mock_mv

        mock_run = MagicMock()
        mock_run.info.artifact_uri = f"file://{tmp_path}/artifacts"
        mock_client.get_run.return_value = mock_run

        tracker = MLflowTracker(experiment_name="test", tracking_uri=str(tmp_path))
        tracker._client = mock_client

        version = tracker.register_model_version(
            run_id="run-abc",
            model_name="git-query-lgbm-reranker",
            artifact_path="models/lgbm.pkl",
        )

        assert version == 3
        mock_client.create_model_version.assert_called_once()
        mock_client.transition_model_version_stage.assert_called_once_with(
            name="git-query-lgbm-reranker", version="3", stage="Staging"
        )

    def test_builds_correct_source_uri(self, tmp_path):
        from src.recommender.mlops.mlflow_tracker import MLFLOW_AVAILABLE, MLflowTracker

        if not MLFLOW_AVAILABLE:
            pytest.skip("MLflow not installed")

        mock_client = MagicMock()
        mock_mv = MagicMock()
        mock_mv.version = "1"
        mock_client.create_model_version.return_value = mock_mv

        mock_run = MagicMock()
        mock_run.info.artifact_uri = "file:///app/mlruns/123/abc/artifacts"
        mock_client.get_run.return_value = mock_run

        tracker = MLflowTracker(experiment_name="test", tracking_uri=str(tmp_path))
        tracker._client = mock_client

        tracker.register_model_version(
            run_id="run-abc",
            model_name="test-model",
            artifact_path="models/lgbm.pkl",
        )

        call_kwargs = mock_client.create_model_version.call_args
        source = call_kwargs.kwargs.get("source") or call_kwargs.args[1]
        assert source == "file:///app/mlruns/123/abc/artifacts/models/lgbm.pkl"

    def test_returns_none_on_client_exception(self, tmp_path):
        from src.recommender.mlops.mlflow_tracker import MLFLOW_AVAILABLE, MLflowTracker

        if not MLFLOW_AVAILABLE:
            pytest.skip("MLflow not installed")

        mock_client = MagicMock()
        mock_client.get_run.side_effect = Exception("run not found")

        tracker = MLflowTracker(experiment_name="test", tracking_uri=str(tmp_path))
        tracker._client = mock_client

        result = tracker.register_model_version(
            run_id="bad-run",
            model_name="test-model",
            artifact_path="models/lgbm.pkl",
        )

        assert result is None


# ---------------------------------------------------------------------------
# P4: get_production_metrics — no production model + normal path
# ---------------------------------------------------------------------------


class TestGetProductionMetrics:
    def test_returns_empty_dict_when_mlflow_unavailable(self):
        from src.recommender.mlops.mlflow_tracker import MLflowTracker

        tracker = MLflowTracker(experiment_name="test")
        tracker._client = None

        result = tracker.get_production_metrics("test-model", ["mean_ndcg_at_10"])
        assert result == {}

    def test_returns_empty_dict_when_no_production_version(self):
        from src.recommender.mlops.mlflow_tracker import MLFLOW_AVAILABLE, MLflowTracker

        if not MLFLOW_AVAILABLE:
            pytest.skip("MLflow not installed")

        mock_client = MagicMock()
        mock_client.get_latest_versions.return_value = []  # no Production version

        tracker = MLflowTracker(experiment_name="test")
        tracker._client = mock_client

        result = tracker.get_production_metrics("git-query-lgbm-reranker", ["mean_ndcg_at_10"])

        assert result == {}

    def test_returns_requested_metrics_from_production_run(self):
        from src.recommender.mlops.mlflow_tracker import MLFLOW_AVAILABLE, MLflowTracker

        if not MLFLOW_AVAILABLE:
            pytest.skip("MLflow not installed")

        mock_client = MagicMock()
        mock_version = MagicMock()
        mock_version.run_id = "prod-run-123"
        mock_client.get_latest_versions.return_value = [mock_version]

        mock_run_data = MagicMock()
        mock_run_data.data.metrics = {
            "mean_ndcg_at_10": 0.82,
            "std_ndcg_at_10": 0.05,
            "unrelated_metric": 99.0,
        }
        mock_client.get_run.return_value = mock_run_data

        tracker = MLflowTracker(experiment_name="test")
        tracker._client = mock_client

        result = tracker.get_production_metrics("git-query-lgbm-reranker", ["mean_ndcg_at_10", "std_ndcg_at_10"])

        assert result == {"mean_ndcg_at_10": 0.82, "std_ndcg_at_10": 0.05}
        assert "unrelated_metric" not in result

    def test_returns_empty_dict_on_client_exception(self):
        from src.recommender.mlops.mlflow_tracker import MLFLOW_AVAILABLE, MLflowTracker

        if not MLFLOW_AVAILABLE:
            pytest.skip("MLflow not installed")

        mock_client = MagicMock()
        mock_client.get_latest_versions.side_effect = Exception("mlflow down")

        tracker = MLflowTracker(experiment_name="test")
        tracker._client = mock_client

        result = tracker.get_production_metrics("test-model", ["mean_ndcg_at_10"])
        assert result == {}
