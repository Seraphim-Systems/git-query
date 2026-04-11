"""Tests for drift monitoring with Evidently AI."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestDriftMonitor:
    """Test suite for DriftMonitor."""

    def test_import_drift_monitor(self):
        """Test that DriftMonitor can be imported."""
        from src.recommender.mlops.drift_monitor import DriftMonitor

        assert DriftMonitor is not None

    def test_monitor_initialization(self):
        """Test monitor initializes with default settings."""
        from src.recommender.mlops.drift_monitor import DriftMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = DriftMonitor(report_dir=tmpdir)
            assert monitor.report_dir == Path(tmpdir)
            assert monitor.report_dir.exists()

    def test_monitor_creates_report_directory(self):
        """Test that monitor creates report directory if it doesn't exist."""
        from src.recommender.mlops.drift_monitor import DriftMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            report_dir = Path(tmpdir) / "nested" / "reports"
            _monitor = DriftMonitor(report_dir=str(report_dir))
            assert report_dir.exists()

    def test_save_report_json(self):
        """Test saving drift report as JSON."""
        from src.recommender.mlops.drift_monitor import DriftMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = DriftMonitor(report_dir=tmpdir)

            drift_info = {
                "timestamp": "2024-01-01T00:00:00",
                "type": "data_drift",
                "drift_detected": False,
                "metrics": {"test": 0.5},
            }

            filepath = monitor.save_report(drift_info, "test_report", format="json")

            assert Path(filepath).exists()
            with open(filepath) as f:
                saved = json.load(f)
            assert saved["type"] == "data_drift"
            assert saved["drift_detected"] is False

    def test_check_embedding_drift_without_evidently(self):
        """Test embedding drift check returns None when Evidently unavailable."""
        from src.recommender.mlops.drift_monitor import EVIDENTLY_AVAILABLE, DriftMonitor

        if EVIDENTLY_AVAILABLE:
            pytest.skip("Evidently is available, testing unavailable case")

        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = DriftMonitor(report_dir=tmpdir)
            ref_emb = np.random.randn(100, 384)
            cur_emb = np.random.randn(100, 384)

            result = monitor.check_embedding_drift(ref_emb, cur_emb)
            assert result is None

    @pytest.mark.skipif(
        not os.getenv("EVIDENTLY_AVAILABLE", "false").lower() == "true", reason="Evidently not available"
    )
    def test_check_embedding_drift_with_evidently(self):
        """Test embedding drift check with Evidently."""
        from src.recommender.mlops.drift_monitor import EVIDENTLY_AVAILABLE, DriftMonitor

        if not EVIDENTLY_AVAILABLE:
            pytest.skip("Evidently not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = DriftMonitor(report_dir=tmpdir)

            # Create reference and current embeddings
            np.random.seed(42)
            ref_emb = np.random.randn(100, 64)  # Use smaller dimension for speed
            cur_emb = np.random.randn(100, 64)

            result = monitor.check_embedding_drift(ref_emb, cur_emb, sample_size=50)

            assert result is not None
            assert "timestamp" in result
            assert "type" in result
            assert result["type"] == "embedding_drift"
            assert "drift_detected" in result

    def test_check_prediction_drift_without_evidently(self):
        """Test prediction drift check returns None when Evidently unavailable."""
        from src.recommender.mlops.drift_monitor import EVIDENTLY_AVAILABLE, DriftMonitor

        if EVIDENTLY_AVAILABLE:
            pytest.skip("Evidently is available, testing unavailable case")

        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = DriftMonitor(report_dir=tmpdir)
            ref_scores = [0.8, 0.7, 0.6, 0.9]
            cur_scores = [0.75, 0.65, 0.55, 0.85]

            result = monitor.check_prediction_drift(ref_scores, cur_scores)
            assert result is None

    def test_extract_drift_status_true(self):
        """Test drift status extraction when drift is detected."""
        from src.recommender.mlops.drift_monitor import DriftMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = DriftMonitor(report_dir=tmpdir)

            # Test with dataset_drift
            result_with_drift = {"metrics": [{"result": {"dataset_drift": True}}]}
            assert monitor._extract_drift_status(result_with_drift) is True

            # Test with drift_detected
            result_with_detected = {"metrics": [{"result": {"drift_detected": True}}]}
            assert monitor._extract_drift_status(result_with_detected) is True

    def test_extract_drift_status_false(self):
        """Test drift status extraction when no drift."""
        from src.recommender.mlops.drift_monitor import DriftMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = DriftMonitor(report_dir=tmpdir)

            result_no_drift = {"metrics": [{"result": {"dataset_drift": False}}]}
            assert monitor._extract_drift_status(result_no_drift) is False

            result_empty = {"metrics": []}
            assert monitor._extract_drift_status(result_empty) is False

    def test_run_full_drift_check_with_none_data(self):
        """Test full drift check handles None inputs gracefully."""
        from src.recommender.mlops.drift_monitor import DriftMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = DriftMonitor(report_dir=tmpdir)

            # Should not raise with all None
            report = monitor.run_full_drift_check()

            assert "timestamp" in report
            assert "checks" in report
            assert "overall_drift_detected" in report
            assert report["overall_drift_detected"] is False

    def test_create_monitor_from_env(self):
        """Test creating monitor from environment variables."""
        from src.recommender.mlops.drift_monitor import create_monitor_from_env

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"EVIDENTLY_REPORT_PATH": tmpdir}):
                monitor = create_monitor_from_env()
                assert str(monitor.report_dir) == tmpdir


class TestExtractDriftStatusFromSnapshot:
    """_extract_drift_status_from_snapshot uses the new Evidently snapshot API."""

    def test_returns_true_when_dataset_drift_attribute_set(self):
        from src.recommender.mlops.drift_monitor import DriftMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = DriftMonitor(report_dir=tmpdir)

            metric_value = MagicMock()
            metric_value.dataset_drift = True
            metric_result = MagicMock()
            metric_result.value = metric_value
            snapshot = MagicMock()
            snapshot.metric_results = [metric_result]

            assert monitor._extract_drift_status_from_snapshot(snapshot) is True

    def test_returns_true_when_drift_detected_attribute_set(self):
        from src.recommender.mlops.drift_monitor import DriftMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = DriftMonitor(report_dir=tmpdir)

            metric_value = MagicMock(spec=[])  # no dataset_drift attribute
            metric_value.drift_detected = True
            metric_result = MagicMock()
            metric_result.value = metric_value
            snapshot = MagicMock()
            snapshot.metric_results = [metric_result]

            assert monitor._extract_drift_status_from_snapshot(snapshot) is True

    def test_returns_false_when_no_drift(self):
        from src.recommender.mlops.drift_monitor import DriftMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = DriftMonitor(report_dir=tmpdir)

            metric_value = MagicMock()
            metric_value.dataset_drift = False
            metric_value.drift_detected = False
            metric_result = MagicMock()
            metric_result.value = metric_value
            snapshot = MagicMock()
            snapshot.metric_results = [metric_result]

            assert monitor._extract_drift_status_from_snapshot(snapshot) is False

    def test_returns_false_on_exception(self):
        """Malformed snapshot does not raise — returns False gracefully."""
        from src.recommender.mlops.drift_monitor import DriftMonitor

        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = DriftMonitor(report_dir=tmpdir)

            snapshot = MagicMock()
            snapshot.metric_results = MagicMock(side_effect=RuntimeError("boom"))

            assert monitor._extract_drift_status_from_snapshot(snapshot) is False


class TestCheckTargetDrift:
    def test_returns_none_when_evidently_unavailable(self):
        from src.recommender.mlops.drift_monitor import EVIDENTLY_AVAILABLE, DriftMonitor

        if EVIDENTLY_AVAILABLE:
            pytest.skip("Evidently is available; testing unavailable path only")

        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = DriftMonitor(report_dir=tmpdir)
            result = monitor.check_target_drift(None, None)

        assert result is None

    def test_returns_error_dict_on_exception(self):
        """When Evidently raises internally, returns {"error": ..., "timestamp": ...}."""
        from src.recommender.mlops.drift_monitor import EVIDENTLY_AVAILABLE, DriftMonitor

        if not EVIDENTLY_AVAILABLE:
            pytest.skip("Evidently not installed")

        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = DriftMonitor(report_dir=tmpdir)

            # Patch Report.run to raise so we hit the except branch
            with patch("evidently.Report.run", side_effect=RuntimeError("forced error")):
                ref = pd.DataFrame({"clicked": [1, 0, 1, 0, 1]})
                cur = pd.DataFrame({"clicked": [0, 0, 0, 1, 0]})
                result = monitor.check_target_drift(ref, cur, target_column="clicked")

        assert "error" in result
        assert "timestamp" in result

    def test_returns_result_dict_with_expected_keys(self):
        """When Evidently works correctly, drift_info has the required keys."""
        from src.recommender.mlops.drift_monitor import EVIDENTLY_AVAILABLE, DriftMonitor

        if not EVIDENTLY_AVAILABLE:
            pytest.skip("Evidently not installed")

        import pandas as pd

        np.random.seed(0)
        ref = pd.DataFrame({"clicked": np.random.randint(0, 2, 50)})
        cur = pd.DataFrame({"clicked": np.random.randint(0, 2, 50)})

        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = DriftMonitor(report_dir=tmpdir)
            result = monitor.check_target_drift(ref, cur, target_column="clicked")

        assert result is not None
        for key in ("timestamp", "type", "reference_ctr", "current_ctr", "ctr_change", "drift_detected"):
            assert key in result, f"Missing key: {key}"
        assert result["type"] == "target_drift"


class TestDriftMonitorWithMockedEvidently:
    """Tests using mocked Evidently components."""

    def test_check_data_drift_mocked(self):
        """Test data drift check with mocked Evidently."""
        from src.recommender.mlops.drift_monitor import EVIDENTLY_AVAILABLE, DriftMonitor

        if not EVIDENTLY_AVAILABLE:
            pytest.skip("Evidently not installed, but mocking requires base import")

        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = DriftMonitor(report_dir=tmpdir)

            # Create simple DataFrames
            import pandas as pd

            ref_data = pd.DataFrame(
                {
                    "stars": [100, 200, 300, 400, 500],
                    "forks": [10, 20, 30, 40, 50],
                }
            )
            cur_data = pd.DataFrame(
                {
                    "stars": [150, 250, 350, 450, 550],
                    "forks": [15, 25, 35, 45, 55],
                }
            )

            result = monitor.check_data_drift(ref_data, cur_data)

            assert result is not None
            assert "timestamp" in result
            assert "type" in result
            assert result["type"] == "data_drift"


class TestDriftMonitorIntegration:
    """Integration tests for drift monitoring."""

    @pytest.mark.integration
    def test_full_workflow(self):
        """Test complete drift monitoring workflow."""
        from src.recommender.mlops.drift_monitor import EVIDENTLY_AVAILABLE, DriftMonitor

        if not EVIDENTLY_AVAILABLE:
            pytest.skip("Evidently not installed")

        import pandas as pd

        with tempfile.TemporaryDirectory() as tmpdir:
            monitor = DriftMonitor(report_dir=tmpdir)

            # Simulate reference and current data
            np.random.seed(42)
            ref_data = pd.DataFrame(
                {
                    "stars": np.random.exponential(1000, 100),
                    "language": np.random.choice(["Python", "JavaScript", "Go"], 100),
                }
            )
            cur_data = pd.DataFrame(
                {
                    "stars": np.random.exponential(1200, 100),  # Slight shift
                    "language": np.random.choice(["Python", "JavaScript", "Go"], 100),
                }
            )

            ref_embeddings = np.random.randn(100, 32)
            cur_embeddings = np.random.randn(100, 32) + 0.1  # Slight shift

            ref_scores = np.random.random(100).tolist()
            cur_scores = (np.random.random(100) + 0.05).tolist()  # Slight shift

            # Run full check
            report = monitor.run_full_drift_check(
                reference_data=ref_data,
                current_data=cur_data,
                reference_embeddings=ref_embeddings,
                current_embeddings=cur_embeddings,
                reference_scores=ref_scores,
                current_scores=cur_scores,
            )

            assert "timestamp" in report
            assert "checks" in report
            assert len(report["checks"]) > 0
            assert "overall_drift_detected" in report

            # Check that reports were saved
            report_files = list(Path(tmpdir).glob("*.json"))
            assert len(report_files) > 0
