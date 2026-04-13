"""Tests for the FastAPI lifespan (startup / shutdown) of the recommender.

Calls the lifespan async context manager directly rather than going through
httpx.AsyncClient — ASGITransport does not trigger ASGI lifespan events, so
the only reliable way to test startup/shutdown behaviour is to drive the
lifespan function itself.
"""

import pytest
from unittest.mock import AsyncMock, patch

from src.recommender.api import app, lifespan
from src.recommender.database import db_manager


# ---------------------------------------------------------------------------
# Shared patch helper
# ---------------------------------------------------------------------------


def _apply_lifespan_patches(mocker):
    """Patch all external I/O that the lifespan triggers.

    Returns (mock_connect, mock_close, mock_embed_load, mock_rerank_load).
    """
    mock_connect = mocker.patch.object(db_manager, "connect", new_callable=AsyncMock)
    mock_close = mocker.patch.object(db_manager, "close", new_callable=AsyncMock)
    mock_embed_load = mocker.patch(
        "src.recommender.services.embedding_service.EmbeddingService.load_active_model",
        new_callable=AsyncMock,
    )
    mock_rerank_load = mocker.patch(
        "src.recommender.services.reranker_service.RerankerService.load_active_model",
        new_callable=AsyncMock,
    )
    mocker.patch(
        "src.recommender.services.embedding_service.EmbeddingService.load_model",
        return_value=None,
    )
    mocker.patch(
        "src.recommender.services.reranker_service.RerankerService.load_model",
        return_value=None,
    )
    mocker.patch(
        "src.recommender.services.registry_service.ModelRegistryService.get_active_model",
        new_callable=AsyncMock,
        return_value=None,
    )
    return mock_connect, mock_close, mock_embed_load, mock_rerank_load


# ---------------------------------------------------------------------------
# Startup tests
# ---------------------------------------------------------------------------


class TestStartup:
    async def test_startup_calls_db_connect(self, mocker):
        """lifespan must call db_manager.connect() exactly once on startup."""
        mock_connect, _, _, _ = _apply_lifespan_patches(mocker)

        async with lifespan(app):
            pass

        mock_connect.assert_awaited_once()

    async def test_startup_creates_all_three_engine_types(self, mocker):
        """app.state.engines must contain baseline, hybrid, and personalized."""
        _apply_lifespan_patches(mocker)

        async with lifespan(app):
            engine_keys = set(app.state.engines.keys())

        assert "baseline" in engine_keys
        assert "hybrid" in engine_keys
        assert "personalized" in engine_keys

    async def test_startup_populates_all_five_services_in_state(self, mocker):
        """app.state must have all five service attributes after startup."""
        _apply_lifespan_patches(mocker)

        async with lifespan(app):
            assert hasattr(app.state, "embedding_service")
            assert hasattr(app.state, "reranker_service")
            assert hasattr(app.state, "personalization_service")
            assert hasattr(app.state, "ab_test_service")
            assert hasattr(app.state, "registry_service")

    async def test_startup_calls_load_active_model_on_embedding_service(self, mocker):
        """load_active_model must be called on the embedding service during startup."""
        _, _, mock_embed_load, _ = _apply_lifespan_patches(mocker)

        async with lifespan(app):
            pass

        mock_embed_load.assert_awaited_once()

    async def test_startup_calls_load_active_model_on_reranker_service(self, mocker):
        """load_active_model must be called on the reranker service during startup."""
        _, _, _, mock_rerank_load = _apply_lifespan_patches(mocker)

        async with lifespan(app):
            pass

        mock_rerank_load.assert_awaited_once()


# ---------------------------------------------------------------------------
# Shutdown tests
# ---------------------------------------------------------------------------


class TestShutdown:
    async def test_shutdown_calls_db_close(self, mocker):
        """lifespan must call db_manager.close() exactly once on shutdown."""
        _, mock_close, _, _ = _apply_lifespan_patches(mocker)

        async with lifespan(app):
            pass  # exits the context → triggers shutdown

        mock_close.assert_awaited_once()
