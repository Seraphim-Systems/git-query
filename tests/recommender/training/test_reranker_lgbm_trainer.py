"""Unit tests for RerankerLGBMTrainer — London School TDD (mock-first)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

_MODULE = "src.recommender.training.trainers.reranker_lgbm_trainer"


def _make_grouped_df():
    """Minimal grouped DataFrame accepted by LGBMRanker."""
    return pd.DataFrame(
        {
            "query_id": [0, 0, 1, 1],
            "query_text": ["python", "python", "javascript", "javascript"],
            "name": ["repo1", "repo2", "repo3", "repo4"],
            "description": ["d1", "d2", "d3", "d4"],
            "language": ["Python", "Python", "JavaScript", "JavaScript"],
            "stars": [100, 50, 200, 10],
        }
    )


def _make_mocks(metrics=None):
    """Return (mock_ranker, mock_registry, mock_settings) tuple."""
    mock_ranker = MagicMock()
    mock_ranker.params = {}
    mock_ranker.train.return_value = metrics or {"mean_ndcg_at_10": 0.8}
    mock_ranker.save.return_value = "/tmp/test-models/lgbm_default.pkl"

    mock_registry = AsyncMock()
    mock_registry.register_model = AsyncMock()
    mock_registry.register_model.return_value = "model-123"

    mock_settings = MagicMock()
    mock_settings.model_path = "/tmp/test-models"

    return mock_ranker, mock_registry, mock_settings


class TestRerankerLGBMTrainer:
    async def test_raises_when_grouped_df_missing(self):
        from src.recommender.training.trainers.reranker_lgbm_trainer import (
            RerankerLGBMTrainer,
        )

        trainer = RerankerLGBMTrainer()
        with pytest.raises(ValueError, match="grouped_df"):
            await trainer.train({}, variant="default")

    async def test_train_calls_lgbm_ranker_train(self):
        from src.recommender.training.trainers.reranker_lgbm_trainer import (
            RerankerLGBMTrainer,
        )

        mock_ranker, mock_registry, mock_settings = _make_mocks()
        training_data = {
            "grouped_df": _make_grouped_df(),
            "dataset_version": "20260101-abc",
        }

        with (
            patch(f"{_MODULE}.LGBMRanker", return_value=mock_ranker),
            patch(f"{_MODULE}.ModelRegistryService", return_value=mock_registry),
            patch(f"{_MODULE}.settings", mock_settings),
        ):
            await RerankerLGBMTrainer().train(training_data, variant="default")

        mock_ranker.train.assert_called_once()

    async def test_train_saves_model(self):
        from src.recommender.training.trainers.reranker_lgbm_trainer import (
            RerankerLGBMTrainer,
        )

        mock_ranker, mock_registry, mock_settings = _make_mocks()
        training_data = {"grouped_df": _make_grouped_df()}

        with (
            patch(f"{_MODULE}.LGBMRanker", return_value=mock_ranker),
            patch(f"{_MODULE}.ModelRegistryService", return_value=mock_registry),
            patch(f"{_MODULE}.settings", mock_settings),
        ):
            await RerankerLGBMTrainer().train(training_data, variant="default")

        mock_ranker.save.assert_called_once()

    async def test_train_uses_explicit_model_dir_when_provided(self):
        from src.recommender.training.trainers.reranker_lgbm_trainer import (
            RerankerLGBMTrainer,
        )

        mock_ranker, mock_registry, mock_settings = _make_mocks()
        training_data = {"grouped_df": _make_grouped_df()}

        with (
            patch(f"{_MODULE}.LGBMRanker", return_value=mock_ranker),
            patch(f"{_MODULE}.ModelRegistryService", return_value=mock_registry),
            patch(f"{_MODULE}.settings", mock_settings),
        ):
            await RerankerLGBMTrainer(model_dir="/tmp/explicit-model-dir").train(
                training_data,
                variant="default",
            )

        save_path = mock_ranker.save.call_args.args[0]
        assert save_path.startswith("/tmp/explicit-model-dir")

    async def test_train_calls_registry_register_model(self):
        from src.recommender.training.trainers.reranker_lgbm_trainer import (
            RerankerLGBMTrainer,
        )

        mock_ranker, mock_registry, mock_settings = _make_mocks()
        training_data = {"grouped_df": _make_grouped_df()}

        with (
            patch(f"{_MODULE}.LGBMRanker", return_value=mock_ranker),
            patch(f"{_MODULE}.ModelRegistryService", return_value=mock_registry),
            patch(f"{_MODULE}.settings", mock_settings),
        ):
            await RerankerLGBMTrainer().train(training_data, variant="default")

        mock_registry.register_model.assert_awaited_once()

    async def test_train_returns_metrics(self):
        from src.recommender.training.trainers.reranker_lgbm_trainer import (
            RerankerLGBMTrainer,
        )

        expected_metrics = {"mean_ndcg_at_10": 0.85, "best_iteration": 200}
        mock_ranker, mock_registry, mock_settings = _make_mocks(
            metrics=expected_metrics
        )
        training_data = {"grouped_df": _make_grouped_df()}

        with (
            patch(f"{_MODULE}.LGBMRanker", return_value=mock_ranker),
            patch(f"{_MODULE}.ModelRegistryService", return_value=mock_registry),
            patch(f"{_MODULE}.settings", mock_settings),
        ):
            result = await RerankerLGBMTrainer().train(training_data, variant="default")

        assert result == {**expected_metrics, "model_id": "model-123"}

    async def test_train_propagates_lgbm_ranker_exception(self):
        """When LGBMRanker.train() raises, the exception propagates from the executor."""
        from src.recommender.training.trainers.reranker_lgbm_trainer import (
            RerankerLGBMTrainer,
        )

        mock_ranker, mock_registry, mock_settings = _make_mocks()
        mock_ranker.train.side_effect = ValueError("bad training data")
        training_data = {"grouped_df": _make_grouped_df()}

        with (
            patch(f"{_MODULE}.LGBMRanker", return_value=mock_ranker),
            patch(f"{_MODULE}.ModelRegistryService", return_value=mock_registry),
            patch(f"{_MODULE}.settings", mock_settings),
        ):
            with pytest.raises(ValueError, match="bad training data"):
                await RerankerLGBMTrainer().train(training_data, variant="default")
