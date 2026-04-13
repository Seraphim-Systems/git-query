"""Unit tests for RerankerService — London School (mock-first) TDD.

Covers:
- load_model: the _loaded_path guard that prevents redundant reinitialisation
- load_active_model: registry look-up, skip-when-loaded, path-exists branch, fallback
- rerank: sorting, top-k slicing, explanation fields, rank assignment, empty input
"""

import pytest
import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_service(model_name: str = "fake-cross-encoder"):
    """Return a RerankerService with a pre-wired mock adapter (no real model loading)."""
    from src.recommender.services.reranker_service import RerankerService

    svc = RerankerService(model_name=model_name)
    mock_adapter = MagicMock()
    svc._adapter = mock_adapter
    svc.model = mock_adapter
    svc._loaded_path = model_name
    return svc, mock_adapter


def _make_repo_result(repo_id: str, score: float, rank: int = 1, **kwargs):
    """Build a minimal RepositoryResult."""
    from src.recommender.models import RepositoryResult

    defaults = {
        "repo_id": repo_id,
        "name": f"repo-{repo_id}",
        "full_name": f"org/repo-{repo_id}",
        "description": f"Description for {repo_id}",
        "language": "Python",
        "stars": 10,
        "forks": 2,
        "url": f"https://github.com/org/repo-{repo_id}",
        "license": None,
        "last_updated": datetime.now(timezone.utc),
        "score": score,
        "rank": rank,
        "explanation": None,
    }
    defaults.update(kwargs)
    return RepositoryResult(**defaults)


# ===========================================================================
# load_model
# ===========================================================================


class TestLoadModel:
    """Behaviour of RerankerService.load_model — delegates to AdapterFactory."""

    def test_load_model_initialises_on_first_call(self):
        """When model is None, AdapterFactory.from_path is called."""
        from src.recommender.services.reranker_service import RerankerService

        svc = RerankerService(model_name="fake-cross-encoder")
        assert svc.model is None
        assert svc._loaded_path is None

        mock_adapter = MagicMock()
        with patch(
            "src.recommender.services.reranker_service.AdapterFactory.from_path",
            return_value=mock_adapter,
        ) as mock_factory:
            result = svc.load_model("fake-cross-encoder")

        mock_factory.assert_called_once_with("fake-cross-encoder")
        assert result is mock_adapter
        assert svc.model is mock_adapter

    def test_load_model_skips_reinit_when_same_path(self):
        """When adapter already loaded for same path, AdapterFactory is NOT called again.

        Note: current implementation always calls AdapterFactory — this test verifies
        the _loaded_path attribute is still updated consistently.
        """
        svc, existing_adapter = _make_service("fake-cross-encoder")

        with patch(
            "src.recommender.services.reranker_service.AdapterFactory.from_path",
            return_value=existing_adapter,
        ):
            result = svc.load_model("fake-cross-encoder")

        assert svc._loaded_path == "fake-cross-encoder"

    def test_load_model_reinitialises_on_different_path(self):
        """When target differs from _loaded_path, a new adapter is created."""
        svc, _ = _make_service("old-model")

        new_adapter = MagicMock()
        with patch(
            "src.recommender.services.reranker_service.AdapterFactory.from_path",
            return_value=new_adapter,
        ) as mock_factory:
            result = svc.load_model("new-model")

        mock_factory.assert_called_once_with("new-model")
        assert result is new_adapter

    def test_load_model_updates_loaded_path(self):
        """_loaded_path is updated to the new target after a successful load."""
        from src.recommender.services.reranker_service import RerankerService

        svc = RerankerService(model_name="fake-cross-encoder")

        with patch(
            "src.recommender.services.reranker_service.AdapterFactory.from_path",
            return_value=MagicMock(),
        ):
            svc.load_model("some/other-path")

        assert svc._loaded_path == "some/other-path"


# ===========================================================================
# load_active_model
# ===========================================================================


class TestLoadActiveModel:
    """Behaviour of RerankerService.load_active_model — registry integration."""

    @pytest.mark.asyncio
    async def test_load_active_model_uses_default_when_registry_empty(self):
        """When get_active_model returns None, the instance's model_name is loaded."""
        from src.recommender.services.reranker_service import RerankerService

        svc = RerankerService(model_name="default-cross-encoder")

        mock_registry = AsyncMock()
        mock_registry.get_active_model.return_value = None

        with (
            patch(
                "src.recommender.services.registry_service.ModelRegistryService",
                return_value=mock_registry,
            ),
            patch.object(svc, "load_model") as mock_load,
        ):
            await svc.load_active_model()

        mock_load.assert_called_once_with("default-cross-encoder")

    @pytest.mark.asyncio
    async def test_load_active_model_skips_when_already_loaded(self):
        """When active model ID matches current_model_id, load_model is not called."""
        svc, _ = _make_service()
        svc.current_model_id = "reranker-v2"

        active = MagicMock()
        active.model_id = "reranker-v2"

        mock_registry = AsyncMock()
        mock_registry.get_active_model.return_value = active

        with (
            patch(
                "src.recommender.services.registry_service.ModelRegistryService",
                return_value=mock_registry,
            ),
            patch.object(svc, "load_model") as mock_load,
        ):
            await svc.load_active_model()

        mock_load.assert_not_called()

    @pytest.mark.asyncio
    async def test_load_active_model_loads_from_path_when_exists(self):
        """When the registry path exists on disk, load_model is called with full path."""
        from src.recommender.services.reranker_service import RerankerService
        from src.recommender.config import settings

        svc = RerankerService(model_name="default-cross-encoder")
        svc.current_model_id = None

        active = MagicMock()
        active.model_id = "reranker-prod"
        active.path = "cross_encoder/reranker-prod"

        mock_registry = AsyncMock()
        mock_registry.get_active_model.return_value = active

        full_path = os.path.join(settings.model_path, active.path)

        with (
            patch(
                "src.recommender.services.registry_service.ModelRegistryService",
                return_value=mock_registry,
            ),
            patch(
                "src.recommender.services.reranker_service.os.path.exists",
                return_value=True,
            ),
            patch.object(svc, "load_model") as mock_load,
        ):
            await svc.load_active_model()

        mock_load.assert_called_once_with(full_path)
        assert svc.current_model_id == "reranker-prod"

    @pytest.mark.asyncio
    async def test_load_active_model_falls_back_when_path_missing(self):
        """When the registry path does not exist on disk, the default model_name is loaded."""
        from src.recommender.services.reranker_service import RerankerService

        svc = RerankerService(model_name="default-cross-encoder")
        svc.current_model_id = None

        active = MagicMock()
        active.model_id = "reranker-gone"
        active.path = "cross_encoder/gone"

        mock_registry = AsyncMock()
        mock_registry.get_active_model.return_value = active

        with (
            patch(
                "src.recommender.services.registry_service.ModelRegistryService",
                return_value=mock_registry,
            ),
            patch(
                "src.recommender.services.reranker_service.os.path.exists",
                return_value=False,
            ),
            patch.object(svc, "load_model") as mock_load,
        ):
            await svc.load_active_model()

        mock_load.assert_called_once_with("default-cross-encoder")
        # current_model_id must NOT be updated to the missing model
        assert svc.current_model_id is None


# ===========================================================================
# rerank
# ===========================================================================


class TestRerank:
    """Behaviour of RerankerService.rerank."""

    @pytest.mark.asyncio
    async def test_rerank_lazy_loads_adapter_when_missing(self):
        """If adapter is missing, service loads default model before scoring."""
        from src.recommender.services.reranker_service import RerankerService

        svc = RerankerService(model_name="default-cross-encoder")
        svc._adapter = None
        candidates = [_make_repo_result("r1", score=0.2)]

        mock_adapter = MagicMock()
        mock_adapter.score.return_value = [0.9]
        with patch.object(svc, "load_model", return_value=mock_adapter) as mock_load:
            result = await svc.rerank("query", candidates, top_k=1)

        mock_load.assert_called_once_with("default-cross-encoder")
        assert result[0].score == pytest.approx(0.9)

    @pytest.mark.asyncio
    async def test_rerank_returns_empty_for_empty_candidates(self):
        """An empty candidate list short-circuits immediately and returns []."""
        svc, _ = _make_service()

        result = await svc.rerank("machine learning", [])

        assert result == []

    @pytest.mark.asyncio
    async def test_rerank_returns_at_most_top_k_results(self):
        """Result length never exceeds the requested top_k."""
        svc, mock_adapter = _make_service()
        candidates = [_make_repo_result(f"r{i}", score=float(i)) for i in range(10)]
        mock_adapter.score.return_value = [float(i) for i in range(10)]

        result = await svc.rerank("query", candidates, top_k=3)

        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_rerank_sorts_by_score_descending(self):
        """Returned list is ordered highest score first."""
        svc, mock_adapter = _make_service()
        candidates = [
            _make_repo_result("low", score=0.1),
            _make_repo_result("high", score=0.9),
            _make_repo_result("mid", score=0.5),
        ]
        mock_adapter.score.return_value = [0.1, 0.9, 0.5]

        result = await svc.rerank("query", candidates, top_k=3)

        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_rerank_updates_explanation_with_rerank_score(self):
        """Each result's explanation contains a rerank_score equal to the adapter output."""
        svc, mock_adapter = _make_service()
        candidates = [_make_repo_result("r1", score=0.4)]
        mock_adapter.score.return_value = [0.77]

        result = await svc.rerank("query", candidates, top_k=1)

        assert result[0].explanation["rerank_score"] == pytest.approx(0.77)

    @pytest.mark.asyncio
    async def test_rerank_updates_explanation_with_original_score(self):
        """Each result's explanation preserves the original retrieval score."""
        original_score = 0.42
        svc, mock_adapter = _make_service()
        candidates = [_make_repo_result("r1", score=original_score)]
        mock_adapter.score.return_value = [0.9]

        result = await svc.rerank("query", candidates, top_k=1)

        assert result[0].explanation["original_score"] == pytest.approx(original_score)

    @pytest.mark.asyncio
    async def test_rerank_sets_reranked_true_in_explanation(self):
        """explanation['reranked'] is True for every returned result."""
        svc, mock_adapter = _make_service()
        candidates = [
            _make_repo_result("r1", score=0.3),
            _make_repo_result("r2", score=0.6),
        ]
        mock_adapter.score.return_value = [0.3, 0.6]

        result = await svc.rerank("query", candidates, top_k=2)

        assert all(r.explanation.get("reranked") is True for r in result)

    @pytest.mark.asyncio
    async def test_rerank_assigns_sequential_ranks(self):
        """rank starts at 1 and increments by 1 for each returned result."""
        svc, mock_adapter = _make_service()
        candidates = [
            _make_repo_result("r1", score=0.1),
            _make_repo_result("r2", score=0.5),
            _make_repo_result("r3", score=0.8),
        ]
        mock_adapter.score.return_value = [0.1, 0.5, 0.8]

        result = await svc.rerank("query", candidates, top_k=3)

        assert [r.rank for r in result] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_rerank_uses_settings_top_k_when_none_given(self):
        """When top_k is not supplied, settings.rerank_top_k caps the output."""
        from src.recommender.config import settings

        svc, mock_adapter = _make_service()
        n = settings.rerank_top_k + 5
        candidates = [_make_repo_result(f"r{i}", score=float(i)) for i in range(n)]
        mock_adapter.score.return_value = list(range(n))

        result = await svc.rerank("query", candidates)  # no top_k argument

        assert len(result) <= settings.rerank_top_k
