"""Unit tests for LGBMRanker and build_query_groups.

Tests that actually train LightGBM are decorated with @requires_lgbm and
skipped automatically when lightgbm is not installed (e.g. local dev without
the full requirements).  All other tests use only imports, tiny DataFrames,
and mock FeatureExtractors — no torch or live services needed.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

try:
    import lightgbm  # noqa: F401

    _LGBM_AVAILABLE = True
except ImportError:
    _LGBM_AVAILABLE = False

requires_lgbm = pytest.mark.skipif(not _LGBM_AVAILABLE, reason="lightgbm not installed")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_grouped_df(n_queries: int = 6, repos_per_query: int = 15, seed: int = 42) -> pd.DataFrame:
    """Minimal grouped DataFrame for training tests."""
    rng = np.random.default_rng(seed)
    rows = []
    for qid in range(n_queries):
        for i in range(repos_per_query):
            rows.append(
                {
                    "query_id": qid,
                    "query_text": f"lang_{qid}",
                    "name": f"repo_{qid}_{i}",
                    "description": "a repo",
                    "stars": int(rng.integers(0, 5000)),
                    "forks": int(rng.integers(0, 500)),
                    "language": f"lang_{qid}",
                    "license": "MIT",
                    "topics": [],
                    "readme": "readme text",
                    "updated_at": "2024-01-01",
                }
            )
    return pd.DataFrame(rows)


def _make_raw_df(n_per_lang: int = 20) -> pd.DataFrame:
    """Raw repo DataFrame (no query_id) for build_query_groups tests."""
    rng = np.random.default_rng(0)
    languages = ["python", "javascript", "go", "rust", "java"]
    rows = []
    for lang in languages:
        for i in range(n_per_lang):
            rows.append(
                {
                    "name": f"{lang}_repo_{i}",
                    "description": "desc",
                    "stars": int(rng.integers(0, 10000)),
                    "forks": int(rng.integers(0, 1000)),
                    "language": lang,
                    "license": "MIT",
                    "topics": [],
                    "readme": "readme",
                    "updated_at": "2024-01-01",
                }
            )
    return pd.DataFrame(rows)


def _mock_feature_extractor(n_features: int = 5) -> MagicMock:
    """FeatureExtractor mock that returns deterministic features per row index.

    Uses the row's positional index as the RNG seed so the same row always
    produces the same features regardless of how many times extract_all has
    been called before (avoids stateful RNG drift across multiple calls).
    """
    fe = MagicMock()

    def extract_all(df: pd.DataFrame, query: str = "") -> pd.DataFrame:
        rows = [np.random.default_rng(i).random(n_features) for i in range(len(df))]
        return pd.DataFrame(rows, columns=[f"feat_{i}" for i in range(n_features)], index=df.index)

    fe.extract_all.side_effect = extract_all
    return fe


# ---------------------------------------------------------------------------
# Import / instantiation (no heavy deps, run without torch)
# ---------------------------------------------------------------------------


class TestImports:
    def test_module_importable(self):
        import importlib.util
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            "_lgbm_test",
            Path(__file__).parents[2] / "src/recommender/training/lgbm_ranker.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "LGBMRanker")
        assert hasattr(mod, "build_query_groups")
        assert hasattr(mod, "DEFAULT_PARAMS")
        assert hasattr(mod, "EVAL_SEEDS")

    def test_default_params_keys(self):
        import importlib.util
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            "_lgbm_test2",
            Path(__file__).parents[2] / "src/recommender/training/lgbm_ranker.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.DEFAULT_PARAMS["objective"] == "lambdarank"
        assert "ndcg" in mod.DEFAULT_PARAMS["metric"]

    def test_eval_seeds(self):
        import importlib.util
        from pathlib import Path

        spec = importlib.util.spec_from_file_location(
            "_lgbm_test3",
            Path(__file__).parents[2] / "src/recommender/training/lgbm_ranker.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert mod.EVAL_SEEDS == [7, 21, 42, 84, 168]


# ---------------------------------------------------------------------------
# build_query_groups (no training — fast)
# ---------------------------------------------------------------------------


class TestBuildQueryGroups:
    def test_returns_dataframe_with_required_cols(self):
        from src.recommender.training.lgbm_ranker import build_query_groups

        raw = _make_raw_df(n_per_lang=20)
        result = build_query_groups(raw, top_n_queries=5, candidates_per_query=15, random_state=42)

        assert isinstance(result, pd.DataFrame)
        assert "query_id" in result.columns
        assert "query_text" in result.columns

    def test_at_least_one_group_created(self):
        from src.recommender.training.lgbm_ranker import build_query_groups

        raw = _make_raw_df(n_per_lang=20)
        result = build_query_groups(raw, top_n_queries=5, random_state=42)
        assert result["query_id"].nunique() >= 1

    def test_raises_on_missing_language_column(self):
        from src.recommender.training.lgbm_ranker import build_query_groups

        df = pd.DataFrame({"stars": [1, 2, 3], "name": ["a", "b", "c"]})
        with pytest.raises(ValueError, match="language"):
            build_query_groups(df)

    def test_raises_when_no_groups_can_be_built(self):
        from src.recommender.training.lgbm_ranker import build_query_groups

        df = pd.DataFrame({"language": ["python"] * 3, "stars": [1, 2, 3]})
        with pytest.raises(ValueError):
            build_query_groups(df, top_n_queries=5)

    def test_query_text_matches_language(self):
        from src.recommender.training.lgbm_ranker import build_query_groups

        raw = _make_raw_df(n_per_lang=20)
        result = build_query_groups(raw, top_n_queries=3, random_state=0)
        for _, grp in result.groupby("query_id"):
            assert grp["query_text"].nunique() == 1

    def test_random_state_produces_reproducible_output(self):
        from src.recommender.training.lgbm_ranker import build_query_groups

        raw = _make_raw_df(n_per_lang=20)
        result1 = build_query_groups(raw, top_n_queries=3, random_state=99)
        result2 = build_query_groups(raw, top_n_queries=3, random_state=99)

        pd.testing.assert_frame_equal(result1.reset_index(drop=True), result2.reset_index(drop=True))


# ---------------------------------------------------------------------------
# LGBMRanker — no-training tests (fast)
# ---------------------------------------------------------------------------


class TestLGBMRankerInit:
    def test_custom_params_merged_with_defaults(self):
        from src.recommender.training.lgbm_ranker import DEFAULT_PARAMS, LGBMRanker

        ranker = LGBMRanker(params={"learning_rate": 0.1, "num_leaves": 63})
        assert ranker.params["learning_rate"] == 0.1
        assert ranker.params["num_leaves"] == 63
        assert ranker.params["objective"] == DEFAULT_PARAMS["objective"]

    def test_predict_raises_before_training(self):
        from src.recommender.training.lgbm_ranker import LGBMRanker

        ranker = LGBMRanker()
        with pytest.raises(RuntimeError, match="not trained"):
            ranker.predict(pd.DataFrame(), feature_extractor=_mock_feature_extractor())

    def test_save_raises_before_training(self, tmp_path):
        from src.recommender.training.lgbm_ranker import LGBMRanker

        ranker = LGBMRanker()
        with pytest.raises(RuntimeError, match="No model to save"):
            ranker.save(tmp_path / "model.pkl")


# ---------------------------------------------------------------------------
# LGBMRanker — training tests (slow, require lightgbm)
# ---------------------------------------------------------------------------


@requires_lgbm
class TestLGBMRankerTraining:
    def test_train_sets_model_and_feature_cols(self):
        from src.recommender.training.lgbm_ranker import LGBMRanker

        grouped = _make_grouped_df(n_queries=6, repos_per_query=15)
        fe = _mock_feature_extractor(n_features=5)

        ranker = LGBMRanker(params={"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [10], "verbose": -1})
        ranker.train(grouped, feature_extractor=fe, seeds=[42], num_boost_rounds=50, early_stopping_rounds=10)

        assert ranker.model is not None
        assert ranker.feature_cols is not None
        assert len(ranker.feature_cols) == 5

    def test_train_returns_expected_metric_keys(self):
        from src.recommender.training.lgbm_ranker import LGBMRanker

        grouped = _make_grouped_df(n_queries=6, repos_per_query=15)
        fe = _mock_feature_extractor()

        ranker = LGBMRanker()
        metrics = ranker.train(grouped, feature_extractor=fe, seeds=[42], num_boost_rounds=30, early_stopping_rounds=5)

        for key in ("mean_ndcg_at_10", "std_ndcg_at_10", "best_iteration", "num_features", "seed_results"):
            assert key in metrics, f"Missing key: {key}"
        assert len(metrics["seed_results"]) == 1
        assert "mean_ndcg_at_10" in metrics["seed_results"][0]

    def test_train_logs_to_mlflow_tracker(self):
        from src.recommender.training.lgbm_ranker import LGBMRanker

        grouped = _make_grouped_df(n_queries=6, repos_per_query=15)
        fe = _mock_feature_extractor()
        mock_tracker = MagicMock()

        ranker = LGBMRanker()
        ranker.train(
            grouped,
            feature_extractor=fe,
            tracker=mock_tracker,
            dataset_version="v_abc123",
            seeds=[42],
            num_boost_rounds=30,
            early_stopping_rounds=5,
        )

        mock_tracker.log_params.assert_called_once()
        mock_tracker.log_metrics.assert_called_once()

        logged_params = mock_tracker.log_params.call_args[0][0]
        assert "dataset_version" in logged_params
        assert logged_params["dataset_version"] == "v_abc123"
        assert "objective" in logged_params

        logged_metrics = mock_tracker.log_metrics.call_args[0][0]
        assert "mean_ndcg_at_10" in logged_metrics

    def test_train_without_holdout_when_fewer_than_five_groups(self):
        """With <5 query groups training completes without a holdout split."""
        from src.recommender.training.lgbm_ranker import LGBMRanker

        grouped = _make_grouped_df(n_queries=3, repos_per_query=15)
        fe = _mock_feature_extractor(n_features=5)

        ranker = LGBMRanker()
        metrics = ranker.train(grouped, feature_extractor=fe, seeds=[42], num_boost_rounds=20)

        assert ranker.model is not None
        assert "mean_ndcg_at_10" in metrics

    def test_train_without_tracker_does_not_raise(self):
        from src.recommender.training.lgbm_ranker import LGBMRanker

        grouped = _make_grouped_df(n_queries=6, repos_per_query=15)
        fe = _mock_feature_extractor()

        ranker = LGBMRanker()
        metrics = ranker.train(grouped, feature_extractor=fe, seeds=[42], num_boost_rounds=20)
        assert "mean_ndcg_at_10" in metrics

    def test_save_load_roundtrip(self, tmp_path):
        from src.recommender.training.lgbm_ranker import LGBMRanker

        grouped = _make_grouped_df(n_queries=6, repos_per_query=15)
        fe = _mock_feature_extractor(n_features=4)

        ranker = LGBMRanker()
        ranker.train(grouped, feature_extractor=fe, seeds=[42], num_boost_rounds=30, early_stopping_rounds=5)

        save_path = tmp_path / "lgbm_test.pkl"
        ranker.save(save_path)
        assert save_path.exists()

        loaded = LGBMRanker.load(save_path)
        assert loaded.feature_cols == ranker.feature_cols
        assert loaded.params == ranker.params

        candidates = _make_grouped_df(n_queries=1, repos_per_query=5)
        scores_orig = ranker.predict(candidates, query_text="test", feature_extractor=fe)
        scores_load = loaded.predict(candidates, query_text="test", feature_extractor=fe)
        np.testing.assert_array_almost_equal(scores_orig, scores_load)

    def test_predict_returns_correct_shape(self):
        from src.recommender.training.lgbm_ranker import LGBMRanker

        grouped = _make_grouped_df(n_queries=6, repos_per_query=15)
        fe = _mock_feature_extractor(n_features=5)

        ranker = LGBMRanker()
        ranker.train(grouped, feature_extractor=fe, seeds=[42], num_boost_rounds=20)

        candidates = _make_grouped_df(n_queries=1, repos_per_query=8)
        scores = ranker.predict(candidates, query_text="python", feature_extractor=fe)
        assert scores.shape == (len(candidates),)

    def test_registry_entry_written(self, tmp_path):
        from src.recommender.training.lgbm_ranker import LGBMRanker

        grouped = _make_grouped_df(n_queries=6, repos_per_query=15)
        fe = _mock_feature_extractor()

        ranker = LGBMRanker()
        ranker.train(grouped, feature_extractor=fe, seeds=[42], num_boost_rounds=20)

        model_path = tmp_path / "model.pkl"
        ranker.save(model_path)
        ranker.save_registry_entry(model_path, tmp_path)

        registry_path = tmp_path / "lgbm_registry_latest.json"
        assert registry_path.exists()

        with open(registry_path) as f:
            reg = json.load(f)

        assert "model_id" in reg
        assert reg["variant"] == "lightgbm_lambdarank"
        assert "feature_cols" in reg
        assert isinstance(reg["feature_cols"], list)


# ---------------------------------------------------------------------------
# _build_rank_data — dot-product label path (no LightGBM required)
# ---------------------------------------------------------------------------


def _make_grouped_df_with_interactions(
    n_queries: int = 3, repos_per_query: int = 12, n_positives: int = 4, seed: int = 42
) -> pd.DataFrame:
    """Grouped DataFrame that includes an interaction_score column."""
    rng = np.random.default_rng(seed)
    rows = []
    for qid in range(n_queries):
        for i in range(repos_per_query):
            score = float(rng.integers(1, 5)) if i < n_positives else 0.0
            rows.append(
                {
                    "query_id": qid,
                    "query_text": f"lang_{qid}",
                    "name": f"repo_{qid}_{i}",
                    "description": "a repo",
                    "stars": int(rng.integers(0, 5000)),
                    "forks": int(rng.integers(0, 500)),
                    "language": f"lang_{qid}",
                    "license": "MIT",
                    "topics": [],
                    "readme": "readme text",
                    "updated_at": "2024-01-01",
                    "interaction_score": score,
                }
            )
    return pd.DataFrame(rows)


class TestBuildRankDataDotProduct:
    def test_labels_are_binary(self):
        """Output labels must be 0/1 regardless of which path is taken."""
        from src.recommender.training.lgbm_ranker import LGBMRanker

        grouped = _make_grouped_df_with_interactions(n_positives=4)
        fe = _mock_feature_extractor(n_features=5)

        ranker = LGBMRanker()
        _, y, groups = ranker._build_rank_data(grouped, fe)

        assert set(y.unique()).issubset({0, 1})
        assert sum(groups) == len(grouped)

    def test_group_sizes_sum_to_row_count(self):
        from src.recommender.training.lgbm_ranker import LGBMRanker

        grouped = _make_grouped_df_with_interactions(n_positives=4)
        fe = _mock_feature_extractor(n_features=5)

        ranker = LGBMRanker()
        _, _, groups = ranker._build_rank_data(grouped, fe)

        assert sum(groups) == len(grouped)

    def test_falls_back_to_star_labels_when_few_positives(self):
        """With <3 positive interactions per group, labels equal star-quartile labels."""
        from src.recommender.training.lgbm_ranker import LGBMRanker

        grouped_with_scores = _make_grouped_df_with_interactions(n_positives=2)
        star_only = grouped_with_scores.drop(columns=["interaction_score"])
        fe = _mock_feature_extractor(n_features=5)

        ranker = LGBMRanker()
        _, y_dot, _ = ranker._build_rank_data(grouped_with_scores, fe)
        _, y_star, _ = ranker._build_rank_data(star_only, fe)

        pd.testing.assert_series_equal(y_dot.reset_index(drop=True), y_star.reset_index(drop=True))

    def test_all_zero_interaction_scores_uses_star_labels(self):
        """All-zero interaction_score → pos_mask empty → star fallback."""
        from src.recommender.training.lgbm_ranker import LGBMRanker

        grouped_zeros = _make_grouped_df_with_interactions(n_positives=0)
        star_only = grouped_zeros.drop(columns=["interaction_score"])
        fe = _mock_feature_extractor(n_features=5)

        ranker = LGBMRanker()
        _, y_dot, _ = ranker._build_rank_data(grouped_zeros, fe)
        _, y_star, _ = ranker._build_rank_data(star_only, fe)

        pd.testing.assert_series_equal(y_dot.reset_index(drop=True), y_star.reset_index(drop=True))

    def test_zero_norm_profile_does_not_raise(self):
        """All-zero feature vectors must not cause ZeroDivisionError."""
        from src.recommender.training.lgbm_ranker import LGBMRanker

        grouped = _make_grouped_df_with_interactions(n_positives=4)

        fe = MagicMock()

        def extract_all_zeros(df: pd.DataFrame, query: str = "") -> pd.DataFrame:
            return pd.DataFrame(
                np.zeros((len(df), 5)),
                columns=[f"feat_{i}" for i in range(5)],
                index=df.index,
            )

        fe.extract_all.side_effect = extract_all_zeros

        ranker = LGBMRanker()
        _, y, groups = ranker._build_rank_data(grouped, fe)  # must not raise

        assert len(y) == len(grouped)

    def test_no_interaction_col_uses_star_labels(self):
        """Without interaction_score column, the dot-product path is never entered."""
        from src.recommender.training.lgbm_ranker import LGBMRanker

        grouped = _make_grouped_df(n_queries=3, repos_per_query=12)
        fe = _mock_feature_extractor(n_features=5)

        ranker = LGBMRanker()
        _, y, groups = ranker._build_rank_data(grouped, fe)

        assert set(y.unique()).issubset({0, 1})
        assert sum(groups) == len(grouped)
