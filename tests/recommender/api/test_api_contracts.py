"""API contract tests for the recommender service.

Uses httpx.AsyncClient with the FastAPI app.  The lifespan is bypassed by
patching db_manager.connect/close to AsyncMock and patching service model
loading so no real infrastructure is needed.
"""

import pytest
import pytest_asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from httpx import AsyncClient, ASGITransport

from src.recommender.api import app
from src.recommender.database import db_manager
from src.recommender.engines.baseline import BaselineEngine


# ---------------------------------------------------------------------------
# Shared fixture — bypasses lifespan and injects mock services
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def client(mocker):
    """AsyncClient with lifespan bypassed (no real DB or model loading)."""
    mocker.patch.object(db_manager, "connect", new_callable=AsyncMock)
    mocker.patch.object(db_manager, "close", new_callable=AsyncMock)

    # Prevent EmbeddingService / RerankerService from loading real models
    mocker.patch(
        "src.recommender.services.embedding_service.EmbeddingService.load_active_model",
        new_callable=AsyncMock,
    )
    mocker.patch(
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

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        # Override engines with mocks so endpoint tests control behaviour
        mock_engine = AsyncMock(spec=BaselineEngine)
        mock_engine.recommend = AsyncMock(return_value=[])
        mock_engine.explain = AsyncMock(
            return_value={"engine": "baseline", "method": "keyword_search", "query": "test"}
        )
        mock_engine.get_metadata = MagicMock(return_value={"name": "baseline", "version": "1.0.0"})
        app.state.engines = {"baseline": mock_engine}
        app.state.ab_test_service = AsyncMock()
        app.state.ab_test_service.get_variant_for_user = AsyncMock(return_value="baseline")
        app.state.personalization_service = AsyncMock()
        yield ac


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class TestHealth:
    async def test_health_returns_200(self, client):
        response = await client.get("/health")
        assert response.status_code == 200

    async def test_health_body_has_status_healthy(self, client):
        response = await client.get("/health")
        assert response.json()["status"] == "healthy"

    async def test_health_body_has_service_recommender(self, client):
        response = await client.get("/health")
        assert response.json()["service"] == "recommender"


# ---------------------------------------------------------------------------
# /recommend
# ---------------------------------------------------------------------------


class TestRecommend:
    async def test_recommend_422_on_missing_query(self, client):
        response = await client.post("/recommend", json={})
        assert response.status_code == 422

    async def test_recommend_422_on_top_k_zero(self, client):
        response = await client.post("/recommend", json={"query": "test", "top_k": 0})
        assert response.status_code == 422

    async def test_recommend_422_on_top_k_above_50(self, client):
        response = await client.post("/recommend", json={"query": "test", "top_k": 51})
        assert response.status_code == 422

    async def test_recommend_200_with_valid_request(self, client):
        response = await client.post("/recommend", json={"query": "machine learning python", "top_k": 5})
        assert response.status_code == 200

    async def test_recommend_response_includes_processing_time(self, client):
        response = await client.post("/recommend", json={"query": "machine learning python", "top_k": 5})
        assert response.status_code == 200
        data = response.json()
        assert "processing_time_ms" in data
        assert isinstance(data["processing_time_ms"], float)

    async def test_recommend_response_includes_variant(self, client):
        response = await client.post("/recommend", json={"query": "machine learning python", "top_k": 5})
        assert response.status_code == 200
        assert "variant" in response.json()

    async def test_recommend_500_when_engine_raises(self, client):
        app.state.engines["baseline"].recommend = AsyncMock(side_effect=RuntimeError("engine blew up"))
        response = await client.post("/recommend", json={"query": "crash test", "top_k": 5})
        assert response.status_code == 500


# ---------------------------------------------------------------------------
# /recommend/explain/{repo_id}
# ---------------------------------------------------------------------------


class TestExplain:
    async def test_explain_200_with_valid_request(self, client):
        # Use a simple repo_id without slashes — %2F decodes to / in the path,
        # which breaks the route match since starlette sees it as a path separator.
        response = await client.post("/recommend/explain/test-repo", json={"query": "test"})
        assert response.status_code == 200
        data = response.json()
        assert "engine" in data
        assert "method" in data

    async def test_explain_500_when_engine_raises(self, client):
        app.state.engines["baseline"].explain = AsyncMock(side_effect=RuntimeError("explode"))
        response = await client.post("/recommend/explain/bad-repo", json={"query": "test"})
        assert response.status_code == 500


# ---------------------------------------------------------------------------
# /interaction
# ---------------------------------------------------------------------------


class TestInteraction:
    async def test_interaction_422_on_invalid_interaction_type(self, client):
        payload = {
            "user_id": "user1",
            "query": "python",
            "repo_id": "repo1",
            "interaction_type": "not_a_valid_type",
            "variant": "baseline",
        }
        response = await client.post("/interaction", json=payload)
        assert response.status_code == 422

    async def test_interaction_200_logs_successfully(self, client, mocker):
        mocker.patch.object(
            db_manager,
            "log_interaction",
            new_callable=AsyncMock,
            return_value="interaction-abc",
        )
        payload = {
            "user_id": "user1",
            "query": "python",
            "repo_id": "repo1",
            "interaction_type": "click",
            "variant": "baseline",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        response = await client.post("/interaction", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["interaction_id"] == "interaction-abc"


# ---------------------------------------------------------------------------
# /preferences/{user_id}
# ---------------------------------------------------------------------------


class TestPreferences:
    async def test_preferences_404_when_not_found(self, client, mocker):
        mocker.patch.object(
            db_manager,
            "get_user_preferences",
            new_callable=AsyncMock,
            return_value=None,
        )
        response = await client.get("/preferences/unknown-user")
        assert response.status_code == 404

    async def test_preferences_200_when_found(self, client, mocker):
        from src.recommender.models import UserPreferences

        prefs = UserPreferences(
            user_id="user1",
            language_preferences={"Python": 0.9},
            topic_preferences={},
            total_interactions=3,
            last_updated=datetime.now(timezone.utc),
        )
        mocker.patch.object(
            db_manager,
            "get_user_preferences",
            new_callable=AsyncMock,
            return_value=prefs,
        )
        response = await client.get("/preferences/user1")
        assert response.status_code == 200
        data = response.json()
        assert data["user_id"] == "user1"


# ---------------------------------------------------------------------------
# /metrics/{variant}
# ---------------------------------------------------------------------------


class TestMetrics:
    async def test_metrics_404_when_not_found(self, client, mocker):
        mocker.patch.object(
            db_manager,
            "get_latest_metrics",
            new_callable=AsyncMock,
            return_value=None,
        )
        response = await client.get("/metrics/nonexistent-variant")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# /ab-test
# ---------------------------------------------------------------------------


class TestABTest:
    async def test_ab_test_returns_no_active_test_when_none(self, client, mocker):
        mocker.patch.object(
            db_manager,
            "get_active_ab_test",
            new_callable=AsyncMock,
            return_value=None,
        )
        response = await client.get("/ab-test")
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") == "no_active_test"


# ---------------------------------------------------------------------------
# /admin/cache/clear
# ---------------------------------------------------------------------------


class TestAdminCache:
    async def test_cache_clear_200(self, mocker):
        # Patch db_manager lifecycle for this test's own client
        mocker.patch.object(db_manager, "connect", new_callable=AsyncMock)
        mocker.patch.object(db_manager, "close", new_callable=AsyncMock)
        mocker.patch(
            "src.recommender.services.embedding_service.EmbeddingService.load_active_model",
            new_callable=AsyncMock,
        )
        mocker.patch(
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

        # Mock the redis client's async scan_iter
        mock_redis = AsyncMock()
        mock_redis.scan_iter = MagicMock(return_value=_aiter([]))
        mocker.patch.object(db_manager, "redis_client", mock_redis)

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            response = await ac.post("/admin/cache/clear")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"


# ---------------------------------------------------------------------------
# /admin/engines
# ---------------------------------------------------------------------------


class TestAdminEngines:
    async def test_engines_list_200(self, client):
        response = await client.get("/admin/engines")
        assert response.status_code == 200
        data = response.json()
        assert "engines" in data
        assert isinstance(data["engines"], list)
        assert len(data["engines"]) >= 1


# ---------------------------------------------------------------------------
# update_user_preferences_task
# ---------------------------------------------------------------------------


class TestUpdateUserPreferencesTask:
    async def test_background_task_updates_prefs_when_repo_found(self, mocker):
        from src.recommender.api import update_user_preferences_task
        from src.recommender.models import UserInteraction, InteractionType

        interaction = UserInteraction(
            user_id="user1",
            query="python",
            repo_id="repo1",
            interaction_type=InteractionType.CLICK,
            variant="baseline",
            timestamp=datetime.now(timezone.utc),
        )
        repo_doc = {"repo_id": "repo1", "name": "test", "language": "Python"}

        mocker.patch.object(
            db_manager,
            "get_repositories_by_repo_ids",
            new_callable=AsyncMock,
            return_value={"repo1": repo_doc},
        )

        mock_personalization = AsyncMock()
        mock_personalization.update_preferences_from_interaction = AsyncMock()

        await update_user_preferences_task(interaction, mock_personalization)

        mock_personalization.update_preferences_from_interaction.assert_awaited_once_with(interaction, repo_doc)

    async def test_background_task_does_nothing_when_repo_not_found(self, mocker):
        from src.recommender.api import update_user_preferences_task
        from src.recommender.models import UserInteraction, InteractionType

        interaction = UserInteraction(
            user_id="user1",
            query="python",
            repo_id="unknown-repo",
            interaction_type=InteractionType.VIEW,
            variant="baseline",
            timestamp=datetime.now(timezone.utc),
        )

        mocker.patch.object(
            db_manager,
            "get_repositories_by_repo_ids",
            new_callable=AsyncMock,
            return_value={},
        )

        mock_personalization = AsyncMock()
        mock_personalization.update_preferences_from_interaction = AsyncMock()

        await update_user_preferences_task(interaction, mock_personalization)

        mock_personalization.update_preferences_from_interaction.assert_not_awaited()

    async def test_background_task_handles_exception_without_raising(self, mocker):
        from src.recommender.api import update_user_preferences_task
        from src.recommender.models import UserInteraction, InteractionType

        interaction = UserInteraction(
            user_id="user1",
            query="python",
            repo_id="repo1",
            interaction_type=InteractionType.SAVE,
            variant="baseline",
            timestamp=datetime.now(timezone.utc),
        )

        mocker.patch.object(
            db_manager,
            "get_repositories_by_repo_ids",
            new_callable=AsyncMock,
            side_effect=RuntimeError("db down"),
        )

        mock_personalization = AsyncMock()

        # Must not raise
        await update_user_preferences_task(interaction, mock_personalization)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _aiter_helper(items):
    for item in items:
        yield item


def _aiter(items):
    """Return an async iterable over items."""
    return _aiter_helper(items)
