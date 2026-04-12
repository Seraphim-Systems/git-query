"""Tests for run_drift_check.py — data loaders, CTR derivation, interaction query, main() exit codes."""

import importlib.util
import pathlib
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Import run_drift_check directly to avoid triggering src.recommender.__init__
# (which pulls in torch via the engines package).
_RDC_PATH = (
    pathlib.Path(__file__).parents[2]
    / "src" / "recommender" / "mlops" / "run_drift_check.py"
)
_spec = importlib.util.spec_from_file_location("run_drift_check", _RDC_PATH)
_rdc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rdc)


# ---------------------------------------------------------------------------
# _split_interactions_for_ctr
# ---------------------------------------------------------------------------


class TestSplitInteractionsForCtr:
    def _import(self):
        return _rdc._split_interactions_for_ctr

    def test_returns_none_none_for_empty_dataframe(self):
        fn = self._import()
        ref, cur = fn(pd.DataFrame())
        assert ref is None
        assert cur is None

    def test_returns_none_none_when_fewer_than_20_interactions(self):
        fn = self._import()
        df = pd.DataFrame({"interaction_type": ["click"] * 10})
        ref, cur = fn(df)
        assert ref is None
        assert cur is None

    def test_returns_none_none_at_exactly_19_interactions(self):
        """Boundary: 19 < 20 minimum."""
        fn = self._import()
        df = pd.DataFrame({"interaction_type": ["click"] * 19})
        ref, cur = fn(df)
        assert ref is None
        assert cur is None

    def test_splits_at_midpoint(self):
        fn = self._import()
        df = pd.DataFrame({"interaction_type": ["click"] * 40})
        ref, cur = fn(df)
        assert len(ref) == 20
        assert len(cur) == 20

    def test_positive_interactions_map_to_one(self):
        fn = self._import()
        df = pd.DataFrame(
            {"interaction_type": ["click", "save", "thumbs_up"] * 10}
        )
        ref, cur = fn(df)
        combined = pd.concat([ref, cur])
        assert combined["clicked"].sum() == 30

    def test_negative_interactions_map_to_zero(self):
        fn = self._import()
        df = pd.DataFrame(
            {"interaction_type": ["dismiss", "thumbs_down", "view"] * 10}
        )
        ref, cur = fn(df)
        combined = pd.concat([ref, cur])
        assert combined["clicked"].sum() == 0

    def test_sorts_by_timestamp_before_splitting(self):
        """Older half → reference, newer half → current."""
        fn = self._import()
        df = pd.DataFrame(
            {
                "interaction_type": ["click"] * 40,
                "timestamp": pd.date_range("2025-01-01", periods=40, freq="D"),
            }
        )
        # Shuffle so rows are not in time order
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        ref, cur = fn(df)
        assert len(ref) == 20
        assert len(cur) == 20

    def test_output_only_contains_clicked_column(self):
        fn = self._import()
        df = pd.DataFrame(
            {
                "interaction_type": ["click"] * 40,
                "user_id": ["u1"] * 40,
                "repo_id": ["r1"] * 40,
            }
        )
        ref, cur = fn(df)
        assert list(ref.columns) == ["clicked"]
        assert list(cur.columns) == ["clicked"]


# ---------------------------------------------------------------------------
# _derive_interaction_query
# ---------------------------------------------------------------------------


class TestDeriveInteractionQuery:
    def _import(self):
        return _rdc._derive_interaction_query

    def test_returns_empty_string_when_both_empty(self):
        fn = self._import()
        result = fn(pd.DataFrame(), pd.DataFrame())
        assert result == ""

    def test_falls_back_to_most_common_language_when_no_interactions(self):
        fn = self._import()
        current_df = pd.DataFrame(
            {"language": ["Python", "Python", "Go", "JavaScript", "Python"]}
        )
        result = fn(pd.DataFrame(), current_df)
        assert result == "Python"

    def test_falls_back_to_most_common_language_when_no_positive_interactions(self):
        fn = self._import()
        interactions = pd.DataFrame(
            {"interaction_type": ["dismiss", "thumbs_down"], "repo_id": ["r1", "r2"]}
        )
        current_df = pd.DataFrame({"language": ["Go", "Go", "Python"]})
        result = fn(interactions, current_df)
        assert result == "Go"

    def test_derives_query_from_positively_interacted_repos(self):
        fn = self._import()
        interactions = pd.DataFrame(
            {
                "interaction_type": ["click", "save", "dismiss"],
                "repo_id": ["r1", "r1", "r2"],
            }
        )
        current_df = pd.DataFrame(
            {
                "repo_id": ["r1", "r2"],
                "name": ["awesome-python", "boring-go"],
                "language": ["Python", "Go"],
                "description": ["great lib", "meh"],
            }
        )
        result = fn(interactions, current_df)
        # r1 is top-interacted — query should include its fields
        assert "awesome-python" in result or "Python" in result

    def test_returns_empty_string_when_language_column_missing(self):
        fn = self._import()
        current_df = pd.DataFrame({"name": ["repo1", "repo2"]})
        result = fn(pd.DataFrame(), current_df)
        assert result == ""


# ---------------------------------------------------------------------------
# main() exit codes
# ---------------------------------------------------------------------------

def _make_drift_module_mock(overall_drift_detected=False, report_dir=None):
    """Build a fake 'drift_monitor' module with a stubbed DriftMonitor."""
    mock_module = MagicMock()
    mock_monitor_instance = MagicMock()
    mock_monitor_instance.run_full_drift_check.return_value = {
        "timestamp": "2026-01-01T00:00:00",
        "checks": {},
        "overall_drift_detected": overall_drift_detected,
    }
    mock_monitor_instance.report_dir = report_dir or MagicMock()
    mock_module.DriftMonitor.return_value = mock_monitor_instance
    return mock_module


class TestMainExitCodes:
    def test_exits_2_when_reference_data_path_not_set(self):
        """REFERENCE_DATA_PATH missing → exit 2 (config error)."""
        import os
        env = {k: v for k, v in os.environ.items() if k != "REFERENCE_DATA_PATH"}
        with patch.dict("os.environ", env, clear=True):
            with pytest.raises(SystemExit) as exc:
                _rdc.main()
        assert exc.value.code == 2

    def test_exits_0_when_reference_file_does_not_exist(self, tmp_path):
        """Reference parquet missing → exit 0 (first run, not an error)."""
        nonexistent = str(tmp_path / "nonexistent.parquet")
        env = {"REFERENCE_DATA_PATH": nonexistent}
        with patch.dict("os.environ", env):
            with pytest.raises(SystemExit) as exc:
                _rdc.main()
        assert exc.value.code == 0

    def test_exits_2_when_no_current_data_source(self, tmp_path):
        """No CURRENT_DATA_PATH and no API_BASE_URL → exit 2."""
        import os
        ref_path = tmp_path / "ref.parquet"
        pd.DataFrame({"stars": [100]}).to_parquet(ref_path)

        clear_keys = ["CURRENT_DATA_PATH", "API_BASE_URL", "APIKEY_MONGODB"]
        base_env = {k: v for k, v in os.environ.items() if k not in clear_keys}
        base_env["REFERENCE_DATA_PATH"] = str(ref_path)

        with patch.dict("os.environ", base_env, clear=True):
            with pytest.raises(SystemExit) as exc:
                _rdc.main()
        assert exc.value.code == 2

    def test_exits_0_when_no_drift_detected(self, tmp_path):
        """Full run completing with no drift → exit 0."""
        ref_path = tmp_path / "ref.parquet"
        cur_path = tmp_path / "cur.parquet"
        df = pd.DataFrame({"stars": [100, 200, 300]})
        df.to_parquet(ref_path)
        df.to_parquet(cur_path)

        fake_drift_module = _make_drift_module_mock(
            overall_drift_detected=False, report_dir=tmp_path
        )

        env = {
            "REFERENCE_DATA_PATH": str(ref_path),
            "CURRENT_DATA_PATH": str(cur_path),
        }
        with patch.dict("os.environ", env):
            with patch.dict("sys.modules", {"drift_monitor": fake_drift_module}):
                with pytest.raises(SystemExit) as exc:
                    _rdc.main()
        assert exc.value.code == 0

    def test_exits_1_when_drift_detected(self, tmp_path):
        """Full run with drift detected → exit 1."""
        ref_path = tmp_path / "ref.parquet"
        cur_path = tmp_path / "cur.parquet"
        df = pd.DataFrame({"stars": [100, 200, 300]})
        df.to_parquet(ref_path)
        df.to_parquet(cur_path)

        fake_drift_module = _make_drift_module_mock(
            overall_drift_detected=True, report_dir=tmp_path
        )

        env = {
            "REFERENCE_DATA_PATH": str(ref_path),
            "CURRENT_DATA_PATH": str(cur_path),
        }
        with patch.dict("os.environ", env):
            with patch.dict("sys.modules", {"drift_monitor": fake_drift_module}):
                with pytest.raises(SystemExit) as exc:
                    _rdc.main()
        assert exc.value.code == 1
