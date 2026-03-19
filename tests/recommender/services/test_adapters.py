"""Unit tests for the reranker adapter package — London School TDD (mock-first)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidate(name: str = "repo", language: str = "Python"):
    c = MagicMock()
    c.name = name
    c.description = f"desc {name}"
    c.language = language
    c.stars = 100
    c.forks = 10
    c.license = "MIT"
    c.topics = []
    c.readme = None
    c.updated_at = None
    c.pushed_at = None
    return c


# ===========================================================================
# AdapterFactory
# ===========================================================================


class TestAdapterFactory:
    def test_pkl_extension_returns_lgbm_adapter(self):
        from src.recommender.services.adapters.adapter_factory import AdapterFactory
        from src.recommender.services.adapters.lgbm_adapter import LGBMAdapter

        with patch.object(LGBMAdapter, "__init__", return_value=None):
            adapter = AdapterFactory.from_path("/models/ranker.pkl")

        assert isinstance(adapter, LGBMAdapter)

    def test_joblib_extension_returns_lgbm_adapter(self):
        from src.recommender.services.adapters.adapter_factory import AdapterFactory
        from src.recommender.services.adapters.lgbm_adapter import LGBMAdapter

        with patch.object(LGBMAdapter, "__init__", return_value=None):
            adapter = AdapterFactory.from_path("/models/ranker.joblib")

        assert isinstance(adapter, LGBMAdapter)

    def test_huggingface_path_returns_cross_encoder_adapter(self):
        from src.recommender.services.adapters.adapter_factory import AdapterFactory
        from src.recommender.services.adapters.cross_encoder_adapter import CrossEncoderAdapter

        with patch.object(CrossEncoderAdapter, "__init__", return_value=None):
            adapter = AdapterFactory.from_path("cross-encoder/ms-marco-MiniLM-L-6-v2")

        assert isinstance(adapter, CrossEncoderAdapter)

    def test_model_name_without_extension_returns_cross_encoder_adapter(self):
        from src.recommender.services.adapters.adapter_factory import AdapterFactory
        from src.recommender.services.adapters.cross_encoder_adapter import CrossEncoderAdapter

        with patch.object(CrossEncoderAdapter, "__init__", return_value=None):
            adapter = AdapterFactory.from_path("sentence-transformers/all-MiniLM-L6-v2")

        assert isinstance(adapter, CrossEncoderAdapter)


# ===========================================================================
# LGBMAdapter
# ===========================================================================


class TestLGBMAdapter:
    def _make_adapter(self, n: int = 3):
        from src.recommender.services.adapters.lgbm_adapter import LGBMAdapter

        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.9, 0.5, 0.1][:n])

        mock_fe = MagicMock()
        import pandas as pd
        mock_fe.extract_all.return_value = pd.DataFrame({"f": [1.0] * n})

        payload = {"model": mock_model, "feature_cols": ["f"]}

        with (
            patch("src.recommender.services.adapters.lgbm_adapter.joblib.load", return_value=payload),
            patch("src.recommender.services.adapters.lgbm_adapter.FeatureExtractor", return_value=mock_fe),
        ):
            adapter = LGBMAdapter("/models/ranker.pkl")

        return adapter, mock_model, mock_fe

    def test_score_returns_list(self):
        adapter, _, _ = self._make_adapter(3)
        candidates = [_make_candidate(f"r{i}") for i in range(3)]
        result = adapter.score("python", candidates)
        assert isinstance(result, list)

    def test_score_length_matches_candidates(self):
        adapter, _, _ = self._make_adapter(3)
        candidates = [_make_candidate(f"r{i}") for i in range(3)]
        result = adapter.score("python", candidates)
        assert len(result) == 3

    def test_score_returns_floats(self):
        adapter, _, _ = self._make_adapter(2)
        candidates = [_make_candidate(f"r{i}") for i in range(2)]
        result = adapter.score("python", candidates)
        assert all(isinstance(s, float) for s in result)

    def test_score_calls_feature_extractor(self):
        adapter, _, mock_fe = self._make_adapter(2)
        candidates = [_make_candidate(f"r{i}") for i in range(2)]
        adapter.score("python", candidates)
        mock_fe.extract_all.assert_called_once()

    def test_score_calls_model_predict(self):
        adapter, mock_model, _ = self._make_adapter(2)
        candidates = [_make_candidate(f"r{i}") for i in range(2)]
        adapter.score("python", candidates)
        mock_model.predict.assert_called_once()


# ===========================================================================
# CrossEncoderAdapter
# ===========================================================================


class TestCrossEncoderAdapter:
    def _make_adapter(self, n: int = 3):
        from src.recommender.services.adapters.cross_encoder_adapter import CrossEncoderAdapter

        mock_ce = MagicMock()
        mock_ce.predict.return_value = np.array([0.9, 0.5, 0.1][:n])

        with patch("src.recommender.services.adapters.cross_encoder_adapter.CrossEncoder", return_value=mock_ce):
            adapter = CrossEncoderAdapter("cross-encoder/ms-marco-MiniLM-L-6-v2")

        return adapter, mock_ce

    def test_score_returns_list(self):
        adapter, _ = self._make_adapter(3)
        candidates = [_make_candidate(f"r{i}") for i in range(3)]
        result = adapter.score("python", candidates)
        assert isinstance(result, list)

    def test_score_length_matches_candidates(self):
        adapter, _ = self._make_adapter(3)
        candidates = [_make_candidate(f"r{i}") for i in range(3)]
        result = adapter.score("python", candidates)
        assert len(result) == 3

    def test_score_returns_floats(self):
        adapter, _ = self._make_adapter(2)
        candidates = [_make_candidate(f"r{i}") for i in range(2)]
        result = adapter.score("python", candidates)
        assert all(isinstance(s, float) for s in result)

    def test_score_calls_model_predict(self):
        adapter, mock_ce = self._make_adapter(2)
        candidates = [_make_candidate(f"r{i}") for i in range(2)]
        adapter.score("python", candidates)
        mock_ce.predict.assert_called_once()
