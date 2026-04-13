"""Tests for ModelPromoter champion/challenger promotion logic."""

import importlib.util
import pathlib
from unittest.mock import MagicMock, patch

import pytest

# Import model_promoter directly so we don't trigger src.recommender.__init__
# (which pulls in torch via the engines package).
_PROMOTER_PATH = pathlib.Path(__file__).parents[2] / "src" / "recommender" / "mlops" / "model_promoter.py"
_spec = importlib.util.spec_from_file_location("model_promoter", _PROMOTER_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ModelPromoter = _mod.ModelPromoter


def _make_tracker(production_metrics=None, raises=False):
    """Return a mock MLflowTracker."""
    tracker = MagicMock()
    if raises:
        tracker.get_production_metrics.side_effect = Exception("mlflow down")
    else:
        tracker.get_production_metrics.return_value = production_metrics or {}
    tracker.log_metrics = MagicMock()
    tracker.transition_model_stage = MagicMock()
    return tracker


def _make_promoter(tracker, recommender_url="http://fake-recommender"):
    return ModelPromoter(recommender_url=recommender_url, mlflow_tracker=tracker)


# ---------------------------------------------------------------------------
# promote_if_better: no production model
# ---------------------------------------------------------------------------


class TestPromoteIfBetterNoProductionModel:
    def test_promotes_unconditionally_when_no_production_model(self):
        """First training run — no production model yet — candidate always promoted."""
        tracker = _make_tracker(production_metrics={})
        promoter = _make_promoter(tracker)

        with patch("requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            mock_post.return_value.raise_for_status = MagicMock()

            promoted = promoter.promote_if_better(
                candidate_model_id="lgbm-v1",
                candidate_metrics={"mean_ndcg_at_10": 0.72},
            )

        assert promoted is True

    def test_promotes_unconditionally_when_no_shared_metrics(self):
        """Production model exists but metrics don't overlap — unconditional promotion."""
        tracker = _make_tracker(production_metrics={"some_other_metric": 0.5})
        promoter = _make_promoter(tracker)

        with patch("requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            mock_post.return_value.raise_for_status = MagicMock()

            promoted = promoter.promote_if_better(
                candidate_model_id="lgbm-v2",
                candidate_metrics={"mean_ndcg_at_10": 0.75},
            )

        assert promoted is True


# ---------------------------------------------------------------------------
# promote_if_better: candidate better than production
# ---------------------------------------------------------------------------


class TestPromoteIfBetterCandidateWins:
    def test_promotes_when_primary_metric_improves(self):
        tracker = _make_tracker(production_metrics={"mean_ndcg_at_10": 0.70})
        promoter = _make_promoter(tracker)

        with patch("requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            mock_post.return_value.raise_for_status = MagicMock()

            promoted = promoter.promote_if_better(
                candidate_model_id="lgbm-v2",
                candidate_metrics={"mean_ndcg_at_10": 0.75},
            )

        assert promoted is True

    def test_transitions_mlflow_version_to_production(self):
        tracker = _make_tracker(production_metrics={"mean_ndcg_at_10": 0.70})
        promoter = _make_promoter(tracker)

        with patch("requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            mock_post.return_value.raise_for_status = MagicMock()

            promoter.promote_if_better(
                candidate_model_id="lgbm-v2",
                candidate_metrics={"mean_ndcg_at_10": 0.75},
                candidate_mlflow_version=3,
            )

        tracker.transition_model_stage.assert_called_once_with("git-query-lgbm-reranker", 3, "Production")

    def test_logs_promotion_blocked_zero_on_success(self):
        tracker = _make_tracker(production_metrics={"mean_ndcg_at_10": 0.70})
        promoter = _make_promoter(tracker)

        with patch("requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            mock_post.return_value.raise_for_status = MagicMock()

            promoter.promote_if_better(
                candidate_model_id="lgbm-v2",
                candidate_metrics={"mean_ndcg_at_10": 0.75},
            )

        tracker.log_metrics.assert_called_with({"promotion_blocked": 0.0})


# ---------------------------------------------------------------------------
# promote_if_better: candidate loses
# ---------------------------------------------------------------------------


class TestPromoteIfBetterCandidateLoses:
    def test_does_not_promote_when_primary_metric_does_not_improve(self):
        tracker = _make_tracker(production_metrics={"mean_ndcg_at_10": 0.80})
        promoter = _make_promoter(tracker)

        promoted = promoter.promote_if_better(
            candidate_model_id="lgbm-v2",
            candidate_metrics={"mean_ndcg_at_10": 0.78},
        )

        assert promoted is False

    def test_does_not_promote_when_primary_metric_equal(self):
        """Strictly greater required — equal is not sufficient."""
        tracker = _make_tracker(production_metrics={"mean_ndcg_at_10": 0.80})
        promoter = _make_promoter(tracker)

        promoted = promoter.promote_if_better(
            candidate_model_id="lgbm-v2",
            candidate_metrics={"mean_ndcg_at_10": 0.80},
        )

        assert promoted is False

    def test_blocked_when_mean_ndcg_degrades_beyond_threshold(self):
        """5% degradation threshold — 10% drop must be blocked."""
        tracker = _make_tracker(production_metrics={"mean_ndcg_at_10": 0.80})
        promoter = _make_promoter(tracker)

        promoted = promoter.promote_if_better(
            candidate_model_id="lgbm-bad",
            candidate_metrics={"mean_ndcg_at_10": 0.70},  # -12.5% from 0.80
        )

        assert promoted is False

    def test_blocked_when_std_ndcg_rises_beyond_threshold(self):
        """Higher std_ndcg is worse — a large rise must be blocked."""
        tracker = _make_tracker(production_metrics={"mean_ndcg_at_10": 0.82, "std_ndcg_at_10": 0.10})
        promoter = _make_promoter(tracker)

        promoted = promoter.promote_if_better(
            candidate_model_id="lgbm-unstable",
            # mean improves but std blows up by >5%
            candidate_metrics={"mean_ndcg_at_10": 0.83, "std_ndcg_at_10": 0.20},
        )

        assert promoted is False

    def test_logs_promotion_blocked_one_when_rejected(self):
        tracker = _make_tracker(production_metrics={"mean_ndcg_at_10": 0.80})
        promoter = _make_promoter(tracker)

        promoter.promote_if_better(
            candidate_model_id="lgbm-bad",
            candidate_metrics={"mean_ndcg_at_10": 0.75},
        )

        tracker.log_metrics.assert_called_with({"promotion_blocked": 1.0})


# ---------------------------------------------------------------------------
# _do_promote: recommender API failures are non-fatal
# ---------------------------------------------------------------------------


class TestDoPromote:
    def test_returns_true_even_when_recommender_unreachable(self):
        """Promotion is still counted as successful if recommender is down."""
        import requests

        tracker = _make_tracker(production_metrics={})
        promoter = _make_promoter(tracker)

        with patch("requests.post", side_effect=requests.RequestException("timeout")):
            promoted = promoter.promote_if_better(
                candidate_model_id="lgbm-v1",
                candidate_metrics={"mean_ndcg_at_10": 0.72},
            )

        assert promoted is True

    def test_skips_mlflow_transition_when_no_version(self):
        """No mlflow_version supplied → transition_model_stage must not be called."""
        tracker = _make_tracker(production_metrics={})
        promoter = _make_promoter(tracker)

        with patch("requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=200)
            mock_post.return_value.raise_for_status = MagicMock()

            promoter.promote_if_better(
                candidate_model_id="lgbm-v1",
                candidate_metrics={"mean_ndcg_at_10": 0.72},
                candidate_mlflow_version=None,
            )

        tracker.transition_model_stage.assert_not_called()
