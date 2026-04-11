"""Unit tests for EmbeddingService — London School (mock-first) TDD.

Covers:
- load_model: the _loaded_path guard that prevents redundant reinitialisation
- load_active_model: registry look-up, skip-when-loaded, path-exists branch, fallback
- embed_text: output shape, async dispatch to thread executor
- embed_batch: output shape, batch_size forwarding
- get_dimension: delegation to model
"""

import asyncio
import inspect
import os
import numpy as np
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service(model_name: str = "fake-model"):
    """Return an EmbeddingService with a pre-wired mock model (no HuggingFace calls)."""
    from src.recommender.services.embedding_service import EmbeddingService

    svc = EmbeddingService(model_name=model_name)
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384
    svc.model = mock_model
    svc._loaded_path = model_name
    return svc, mock_model


def _make_repo_result(**kwargs):
    """Build a minimal RepositoryResult with required fields."""
    from src.recommender.models import RepositoryResult

    defaults = {
        "repo_id": "r1",
        "name": "alpha",
        "full_name": "org/alpha",
        "description": "Alpha project",
        "language": "Python",
        "stars": 42,
        "forks": 7,
        "url": "https://github.com/org/alpha",
        "license": None,
        "last_updated": datetime.now(timezone.utc),
        "score": 0.9,
        "rank": 1,
        "explanation": None,
    }
    defaults.update(kwargs)
    return RepositoryResult(**defaults)


# ===========================================================================
# load_model
# ===========================================================================


class TestLoadModel:
    """Behaviour of EmbeddingService.load_model — specifically the _loaded_path guard."""

    def test_load_model_initialises_on_first_call(self):
        """When model is None and _loaded_path is None, SentenceTransformer is created."""
        from src.recommender.services.embedding_service import EmbeddingService

        svc = EmbeddingService(model_name="fake-model")
        assert svc.model is None
        assert svc._loaded_path is None

        mock_instance = MagicMock()
        with patch(
            "src.recommender.services.embedding_service.SentenceTransformer",
            return_value=mock_instance,
        ) as MockST:
            result = svc.load_model("fake-model")

        MockST.assert_called_once_with("fake-model", device=svc.device)
        assert result is mock_instance
        assert svc.model is mock_instance

    def test_load_model_skips_reinit_when_same_path(self):
        """When model is already loaded and target matches _loaded_path, no new instance."""
        from src.recommender.services.embedding_service import EmbeddingService

        svc, existing_model = _make_service("fake-model")

        with patch(
            "src.recommender.services.embedding_service.SentenceTransformer"
        ) as MockST:
            result = svc.load_model("fake-model")

        MockST.assert_not_called()
        assert result is existing_model

    def test_load_model_reinitialises_on_different_path(self):
        """When target differs from _loaded_path, SentenceTransformer is recreated."""
        from src.recommender.services.embedding_service import EmbeddingService

        svc, _ = _make_service("old-model")

        new_instance = MagicMock()
        with patch(
            "src.recommender.services.embedding_service.SentenceTransformer",
            return_value=new_instance,
        ) as MockST:
            result = svc.load_model("new-model")

        MockST.assert_called_once_with("new-model", device=svc.device)
        assert result is new_instance

    def test_load_model_updates_loaded_path_after_init(self):
        """_loaded_path is updated to the new target after a successful load."""
        from src.recommender.services.embedding_service import EmbeddingService

        svc = EmbeddingService(model_name="fake-model")

        with patch(
            "src.recommender.services.embedding_service.SentenceTransformer",
            return_value=MagicMock(),
        ):
            svc.load_model("new-path/model")

        assert svc._loaded_path == "new-path/model"


# ===========================================================================
# load_active_model
# ===========================================================================


class TestLoadActiveModel:
    """Behaviour of EmbeddingService.load_active_model — registry integration."""

    @pytest.mark.asyncio
    async def test_load_active_model_uses_default_when_registry_empty(self):
        """When get_active_model returns None, the instance's model_name is loaded."""
        from src.recommender.services.embedding_service import EmbeddingService

        svc = EmbeddingService(model_name="default-embed-model")

        mock_registry = AsyncMock()
        mock_registry.get_active_model.return_value = None

        with patch(
            "src.recommender.services.registry_service.ModelRegistryService",  # noqa: SIM117
            return_value=mock_registry,
        ) as _MockReg, patch.object(svc, "load_model") as mock_load:
            await svc.load_active_model()

        mock_load.assert_called_once_with("default-embed-model")

    @pytest.mark.asyncio
    async def test_load_active_model_skips_when_already_loaded(self):
        """When active model ID matches current_model_id, load_model is not called."""
        from src.recommender.services.embedding_service import EmbeddingService

        svc, _ = _make_service()
        svc.current_model_id = "model-abc"

        active = MagicMock()
        active.model_id = "model-abc"

        mock_registry = AsyncMock()
        mock_registry.get_active_model.return_value = active

        with patch(
            "src.recommender.services.registry_service.ModelRegistryService",
            return_value=mock_registry,
        ), patch.object(svc, "load_model") as mock_load:
            await svc.load_active_model()

        mock_load.assert_not_called()

    @pytest.mark.asyncio
    async def test_load_active_model_loads_from_path_when_exists(self):
        """When the registry path exists on disk, load_model is called with the full path."""
        from src.recommender.services.embedding_service import EmbeddingService
        from src.recommender.config import settings

        svc = EmbeddingService(model_name="default-embed-model")
        svc.current_model_id = None

        active = MagicMock()
        active.model_id = "model-xyz"
        active.path = "embeddings/model-xyz"

        mock_registry = AsyncMock()
        mock_registry.get_active_model.return_value = active

        full_path = os.path.join(settings.model_path, active.path)

        with patch(
            "src.recommender.services.registry_service.ModelRegistryService",
            return_value=mock_registry,
        ), patch(
            "src.recommender.services.embedding_service.os.path.exists",
            return_value=True,
        ), patch.object(
            svc, "load_model"
        ) as mock_load:
            await svc.load_active_model()

        mock_load.assert_called_once_with(full_path)
        assert svc.current_model_id == "model-xyz"

    @pytest.mark.asyncio
    async def test_load_active_model_falls_back_when_path_missing(self):
        """When the registry path does not exist, the default model_name is loaded instead."""
        from src.recommender.services.embedding_service import EmbeddingService

        svc = EmbeddingService(model_name="default-embed-model")
        svc.current_model_id = None

        active = MagicMock()
        active.model_id = "model-missing"
        active.path = "embeddings/model-missing"

        mock_registry = AsyncMock()
        mock_registry.get_active_model.return_value = active

        with patch(
            "src.recommender.services.registry_service.ModelRegistryService",
            return_value=mock_registry,
        ), patch(
            "src.recommender.services.embedding_service.os.path.exists",
            return_value=False,
        ), patch.object(
            svc, "load_model"
        ) as mock_load:
            await svc.load_active_model()

        mock_load.assert_called_once_with("default-embed-model")
        # current_model_id must NOT be updated to the missing model
        assert svc.current_model_id is None


# ===========================================================================
# embed_text
# ===========================================================================


class TestEmbedText:
    """Behaviour of EmbeddingService.embed_text."""

    @pytest.mark.asyncio
    async def test_embed_text_returns_list_of_floats(self):
        """embed_text returns a plain Python list of floats."""
        svc, mock_model = _make_service()
        mock_model.encode.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)

        result = await svc.embed_text("python web framework")

        assert isinstance(result, list)
        assert all(isinstance(v, float) for v in result)

    @pytest.mark.asyncio
    async def test_embed_text_uses_thread_executor(self):
        """embed_text is a coroutine — it does not block the event loop directly."""
        svc, mock_model = _make_service()
        mock_model.encode.return_value = np.zeros(384, dtype=np.float32)

        coro = svc.embed_text("test text")
        assert inspect.iscoroutine(coro), "embed_text must return a coroutine"
        await coro  # consume it so no ResourceWarning

    @pytest.mark.asyncio
    async def test_embed_text_output_length_matches_dimension(self):
        """Output list length equals the vector dimension returned by the mock model."""
        dim = 384
        svc, mock_model = _make_service()
        mock_model.encode.return_value = np.zeros(dim, dtype=np.float32)

        result = await svc.embed_text("some query")

        assert len(result) == dim


# ===========================================================================
# embed_batch
# ===========================================================================


class TestEmbedBatch:
    """Behaviour of EmbeddingService.embed_batch."""

    @pytest.mark.asyncio
    async def test_embed_batch_returns_list_of_embeddings(self):
        """embed_batch returns a list-of-lists (one per input text)."""
        texts = ["alpha", "beta", "gamma"]
        svc, mock_model = _make_service()
        mock_model.encode.return_value = np.zeros((3, 384), dtype=np.float32)

        result = await svc.embed_batch(texts)

        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(row, list) for row in result)

    @pytest.mark.asyncio
    async def test_embed_batch_uses_configured_batch_size(self):
        """_embed_batch_sync forwards settings.batch_size to model.encode."""
        from src.recommender.config import settings

        texts = ["a", "b"]
        svc, mock_model = _make_service()
        mock_model.encode.return_value = np.zeros((2, 384), dtype=np.float32)

        await svc.embed_batch(texts)

        _, kwargs = mock_model.encode.call_args
        assert kwargs.get("batch_size") == settings.batch_size


# ===========================================================================
# get_dimension
# ===========================================================================


class TestGetDimension:
    """Behaviour of EmbeddingService.get_dimension."""

    def test_get_dimension_returns_integer(self):
        """get_dimension calls model.get_sentence_embedding_dimension and returns int."""
        svc, mock_model = _make_service()
        mock_model.get_sentence_embedding_dimension.return_value = 384

        dim = svc.get_dimension()

        mock_model.get_sentence_embedding_dimension.assert_called_once()
        assert isinstance(dim, int)
        assert dim == 384
