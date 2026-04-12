"""Unit tests for EmbeddingIndexingPipeline — London School TDD."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODULE = "src.recommender.training.pipelines.embedding_indexing_pipeline"

_PIPELINE_DEFAULTS = dict(
    api_base_url="http://api.example.com",
    api_key="test-key",
    models_dir="/models",
    data_cache_dir="/cache",
)


def _make_pipeline(**overrides):
    from src.recommender.training.pipelines.embedding_indexing_pipeline import (
        EmbeddingIndexingPipeline,
    )

    kwargs = {**_PIPELINE_DEFAULTS, **overrides}
    return EmbeddingIndexingPipeline(**kwargs)


def _mock_loop(run_result=None):
    """Return a mock event loop whose run_in_executor is an AsyncMock."""
    loop = MagicMock()
    loop.run_in_executor = AsyncMock(return_value=run_result)
    return loop


# ===========================================================================
# run() — chunked vs non-chunked dispatch
# ===========================================================================


class TestEmbeddingIndexingPipelineRun:
    async def test_run_calls_run_chunked_by_default(self):
        pipeline = _make_pipeline()
        mock_indexer_instance = MagicMock()
        mock_indexer_instance.run_chunked = MagicMock()
        mock_indexer_instance.run = MagicMock()
        loop = _mock_loop()

        with (
            patch(f"{_MODULE}._EmbeddingIndexer", return_value=mock_indexer_instance),
            patch("asyncio.get_running_loop", return_value=loop),
        ):
            await pipeline.run()

        assert loop.run_in_executor.called
        call_args = loop.run_in_executor.call_args
        # The callable passed to run_in_executor should wrap run_chunked, not run
        fn = call_args[0][1]
        fn()
        mock_indexer_instance.run_chunked.assert_called_once()
        mock_indexer_instance.run.assert_not_called()

    async def test_run_calls_run_when_use_chunked_false(self):
        pipeline = _make_pipeline(use_chunked=False)
        mock_indexer_instance = MagicMock()
        mock_indexer_instance.run_chunked = MagicMock()
        mock_indexer_instance.run = MagicMock()
        loop = _mock_loop()

        with (
            patch(f"{_MODULE}._EmbeddingIndexer", return_value=mock_indexer_instance),
            patch("asyncio.get_running_loop", return_value=loop),
        ):
            await pipeline.run()

        assert loop.run_in_executor.called
        call_args = loop.run_in_executor.call_args
        fn = call_args[0][1]
        fn()
        mock_indexer_instance.run.assert_called_once()
        mock_indexer_instance.run_chunked.assert_not_called()


# ===========================================================================
# run() — parameter forwarding
# ===========================================================================


class TestEmbeddingIndexingPipelineParams:
    async def test_run_passes_model_name(self):
        pipeline = _make_pipeline(model_name="all-MiniLM-L6-v2")
        mock_indexer_instance = MagicMock()
        mock_indexer_instance.run_chunked = MagicMock()
        loop = _mock_loop()

        with (
            patch(f"{_MODULE}._EmbeddingIndexer", return_value=mock_indexer_instance),
            patch("asyncio.get_running_loop", return_value=loop),
        ):
            await pipeline.run()

        fn = loop.run_in_executor.call_args[0][1]
        fn()
        call_kwargs = mock_indexer_instance.run_chunked.call_args
        assert call_kwargs is not None
        args, kwargs = call_kwargs
        assert "all-MiniLM-L6-v2" in args or kwargs.get("model_name") == "all-MiniLM-L6-v2"

    async def test_run_passes_max_repos(self):
        pipeline = _make_pipeline(max_repos=500)
        mock_indexer_instance = MagicMock()
        mock_indexer_instance.run_chunked = MagicMock()
        loop = _mock_loop()

        with (
            patch(f"{_MODULE}._EmbeddingIndexer", return_value=mock_indexer_instance),
            patch("asyncio.get_running_loop", return_value=loop),
        ):
            await pipeline.run()

        fn = loop.run_in_executor.call_args[0][1]
        fn()
        call_kwargs = mock_indexer_instance.run_chunked.call_args
        args, kwargs = call_kwargs
        assert 500 in args or kwargs.get("max_repos") == 500

    async def test_indexer_constructed_with_correct_params(self):
        pipeline = _make_pipeline(
            api_base_url="http://custom-api.example.com",
            api_key="secret-key",
            models_dir="/custom/models",
            data_cache_dir="/custom/cache",
        )
        loop = _mock_loop()

        with (
            patch(f"{_MODULE}._EmbeddingIndexer") as mock_indexer_cls,
            patch("asyncio.get_running_loop", return_value=loop),
        ):
            mock_indexer_cls.return_value = MagicMock()
            mock_indexer_cls.return_value.run_chunked = MagicMock()
            await pipeline.run()

        mock_indexer_cls.assert_called_once()
        call_args, call_kwargs = mock_indexer_cls.call_args
        all_kwargs = {
            **dict(zip(["api_base_url", "api_key", "models_dir", "data_cache_dir"], call_args)),
            **call_kwargs,
        }
        assert all_kwargs.get("api_base_url") == "http://custom-api.example.com"
        assert all_kwargs.get("api_key") == "secret-key"
        assert all_kwargs.get("models_dir") == "/custom/models"
        assert all_kwargs.get("data_cache_dir") == "/custom/cache"


# ===========================================================================
# Lifecycle stubs — fetch, train, evaluate, register
# ===========================================================================


class TestEmbeddingIndexingPipelineLifecycleStubs:
    async def test_fetch_raises_not_implemented(self):
        pipeline = _make_pipeline()
        with pytest.raises(NotImplementedError):
            await pipeline.fetch()

    async def test_train_raises_not_implemented(self):
        pipeline = _make_pipeline()
        with pytest.raises(NotImplementedError):
            await pipeline.train({})

    async def test_evaluate_returns_metrics_unchanged(self):
        pipeline = _make_pipeline()
        metrics = {"ndcg": 0.75, "precision": 0.9}
        result = await pipeline.evaluate({}, metrics)
        assert result == metrics

    async def test_register_returns_none(self):
        pipeline = _make_pipeline()
        result = await pipeline.register({"ndcg": 0.75})
        assert result is None
