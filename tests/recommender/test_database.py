"""Unit tests for DatabaseManager — London School TDD.

All tests inject mocks directly into instance attributes on the module-level
`db_manager` singleton; `connect()` is never called.

asyncio_mode = auto (configured in pytest.ini or pyproject.toml) means
@pytest.mark.asyncio is not required.
"""

import asyncio
import json
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

# ---------------------------------------------------------------------------
# Async cursor helper — simulates Motor's find() cursor
# ---------------------------------------------------------------------------


def async_cursor(docs):
    class _Cursor:
        def __init__(self, items):
            self._items = iter(items)

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._items)
            except StopIteration:
                raise StopAsyncIteration

        def sort(self, *a, **k):
            return self

        def limit(self, n):
            return self

        def skip(self, n):
            return self

    return _Cursor(docs)


# ---------------------------------------------------------------------------
# Shared fixture: patch the module-level db_manager's attributes
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_db_manager():
    """Ensure each test starts with a clean db_manager state."""
    from src.recommender.database import db_manager

    original_db = db_manager.db
    original_mongo = db_manager.mongo_client
    original_qdrant = db_manager.qdrant_client
    original_redis = db_manager.redis_client

    yield

    db_manager.db = original_db
    db_manager.mongo_client = original_mongo
    db_manager.qdrant_client = original_qdrant
    db_manager.redis_client = original_redis


# ---------------------------------------------------------------------------
# Tests: connect / close
# ---------------------------------------------------------------------------


class TestConnect:
    async def test_connect_initialises_all_three_clients(self, mocker):
        from src.recommender.database import DatabaseManager

        mock_motor_cls = mocker.patch("src.recommender.database.AsyncIOMotorClient")
        mock_qdrant_cls = mocker.patch("src.recommender.database.QdrantClient")
        mock_redis_from_url = mocker.patch("src.recommender.database.redis.from_url", new_callable=AsyncMock)
        mocker.patch.object(DatabaseManager, "_ensure_collections", new_callable=AsyncMock)

        mock_motor_instance = MagicMock()
        mock_motor_instance.__getitem__ = MagicMock(return_value=MagicMock())
        mock_motor_cls.return_value = mock_motor_instance

        mock_qdrant_instance = MagicMock()
        mock_qdrant_cls.return_value = mock_qdrant_instance

        mock_redis_instance = AsyncMock()
        mock_redis_from_url.return_value = mock_redis_instance

        manager = DatabaseManager()
        await manager.connect()

        mock_motor_cls.assert_called_once()
        mock_qdrant_cls.assert_called_once()
        mock_redis_from_url.assert_called_once()

        assert manager.mongo_client is mock_motor_instance
        assert manager.qdrant_client is mock_qdrant_instance
        assert manager.redis_client is mock_redis_instance


class TestClose:
    async def test_close_closes_mongo_and_redis(self):
        from src.recommender.database import db_manager

        mock_mongo = MagicMock()
        mock_redis = AsyncMock()

        db_manager.mongo_client = mock_mongo
        db_manager.redis_client = mock_redis

        await db_manager.close()

        mock_mongo.close.assert_called_once()
        mock_redis.close.assert_called_once()

    async def test_close_is_safe_when_clients_are_none(self):
        from src.recommender.database import db_manager

        db_manager.mongo_client = None
        db_manager.redis_client = None

        # Must not raise
        await db_manager.close()


# ---------------------------------------------------------------------------
# Tests: log_interaction
# ---------------------------------------------------------------------------


class TestLogInteraction:
    async def test_log_interaction_inserts_and_returns_id(self):
        from src.recommender.database import db_manager
        from src.recommender.models import UserInteraction, InteractionType

        fake_id = "507f1f77bcf86cd799439011"
        mock_collection = AsyncMock()
        mock_collection.insert_one.return_value = MagicMock(inserted_id=fake_id)

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        db_manager.db = mock_db

        interaction = UserInteraction(
            user_id="u1",
            query="python web",
            repo_id="r1",
            interaction_type=InteractionType.CLICK,
            timestamp=datetime.now(timezone.utc),
        )

        result = await db_manager.log_interaction(interaction)

        mock_collection.insert_one.assert_called_once()
        assert result == str(fake_id)

    async def test_log_interaction_serialises_model(self):
        from src.recommender.database import db_manager
        from src.recommender.models import UserInteraction, InteractionType

        mock_collection = AsyncMock()
        mock_collection.insert_one.return_value = MagicMock(inserted_id="abc")

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        db_manager.db = mock_db

        interaction = UserInteraction(
            user_id="u2",
            query="rust async",
            repo_id="r2",
            interaction_type=InteractionType.SAVE,
            timestamp=datetime.now(timezone.utc),
        )

        await db_manager.log_interaction(interaction)

        inserted_doc = mock_collection.insert_one.call_args[0][0]
        assert inserted_doc["user_id"] == "u2"
        assert inserted_doc["repo_id"] == "r2"
        assert inserted_doc["interaction_type"] == InteractionType.SAVE


# ---------------------------------------------------------------------------
# Tests: get_user_interactions
# ---------------------------------------------------------------------------


class TestGetUserInteractions:
    async def test_get_user_interactions_returns_list(self):
        from src.recommender.database import db_manager
        from src.recommender.models import InteractionType

        now = datetime.now(timezone.utc)
        doc1 = {
            "user_id": "u1",
            "query": "q",
            "repo_id": "r1",
            "interaction_type": InteractionType.CLICK,
            "variant": "baseline",
            "timestamp": now,
            "metadata": {},
        }
        doc2 = {
            "user_id": "u1",
            "query": "q2",
            "repo_id": "r2",
            "interaction_type": InteractionType.VIEW,
            "variant": "baseline",
            "timestamp": now,
            "metadata": {},
        }

        mock_collection = MagicMock()
        mock_collection.find.return_value = async_cursor([doc1, doc2])

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        db_manager.db = mock_db

        results = await db_manager.get_user_interactions("u1")

        assert len(results) == 2
        assert results[0].user_id == "u1"
        assert results[1].repo_id == "r2"

    async def test_get_user_interactions_empty_when_no_docs(self):
        from src.recommender.database import db_manager

        mock_collection = MagicMock()
        mock_collection.find.return_value = async_cursor([])

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        db_manager.db = mock_db

        results = await db_manager.get_user_interactions("u_nobody")

        assert results == []

    async def test_get_user_interactions_strips_id_field(self):
        from src.recommender.database import db_manager
        from src.recommender.models import InteractionType

        now = datetime.now(timezone.utc)
        doc = {
            "_id": "mongo_object_id",
            "user_id": "u1",
            "query": "q",
            "repo_id": "r1",
            "interaction_type": InteractionType.CLICK,
            "variant": "baseline",
            "timestamp": now,
            "metadata": {},
        }

        mock_collection = MagicMock()
        mock_collection.find.return_value = async_cursor([doc])

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        db_manager.db = mock_db

        results = await db_manager.get_user_interactions("u1")

        # UserInteraction must be constructable without _id (it would raise if present)
        assert len(results) == 1
        assert not hasattr(results[0], "_id")


# ---------------------------------------------------------------------------
# Tests: get_user_preferences
# ---------------------------------------------------------------------------


class TestGetUserPreferences:
    async def test_get_user_preferences_returns_preferences_when_found(self):
        from src.recommender.database import db_manager

        doc = {
            "user_id": "u1",
            "language_preferences": {"Python": 0.8},
            "topic_preferences": {"async": 0.6},
            "total_interactions": 10,
            "last_updated": datetime.now(timezone.utc),
        }

        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = doc

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        db_manager.db = mock_db

        prefs = await db_manager.get_user_preferences("u1")

        assert prefs is not None
        assert prefs.user_id == "u1"
        assert prefs.language_preferences["Python"] == 0.8

    async def test_get_user_preferences_returns_none_when_not_found(self):
        from src.recommender.database import db_manager

        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = None

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        db_manager.db = mock_db

        prefs = await db_manager.get_user_preferences("u_missing")

        assert prefs is None


# ---------------------------------------------------------------------------
# Tests: update_user_preferences
# ---------------------------------------------------------------------------


class TestUpdateUserPreferences:
    async def test_update_user_preferences_calls_update_one_with_upsert(self):
        from src.recommender.database import db_manager
        from src.recommender.models import UserPreferences

        mock_collection = AsyncMock()

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        db_manager.db = mock_db

        prefs = UserPreferences(
            user_id="u1",
            language_preferences={"Go": 0.9},
            topic_preferences={},
            total_interactions=3,
            last_updated=datetime.now(timezone.utc),
        )

        await db_manager.update_user_preferences(prefs)

        mock_collection.update_one.assert_called_once()
        call_kwargs = mock_collection.update_one.call_args
        assert call_kwargs[1]["upsert"] is True
        filter_arg = call_kwargs[0][0]
        assert filter_arg == {"user_id": "u1"}


# ---------------------------------------------------------------------------
# Tests: search_repositories
# ---------------------------------------------------------------------------


class TestSearchRepositories:
    async def test_search_repositories_returns_from_first_collection_when_populated(self):
        from src.recommender.database import db_manager

        doc = {"_id": "abc123", "name": "my-repo"}

        populated_collection = MagicMock()
        populated_collection.find.return_value = async_cursor([doc])

        empty_collection = MagicMock()
        empty_collection.find.return_value = async_cursor([])

        def _getitem(name):
            from src.recommender.config import settings

            if name == settings.repos_collection:
                return populated_collection
            return empty_collection

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(side_effect=_getitem)
        db_manager.db = mock_db

        results = await db_manager.search_repositories({})

        assert len(results) == 1
        # raw_repos collection must not be queried
        empty_collection.find.assert_not_called()

    async def test_search_repositories_returns_empty_when_repos_collection_empty(self):
        from src.recommender.database import db_manager

        empty_collection = MagicMock()
        empty_collection.find.return_value = async_cursor([])

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=empty_collection)
        db_manager.db = mock_db

        results = await db_manager.search_repositories({})

        assert results == []

    async def test_search_repositories_returns_empty_when_both_empty(self):
        from src.recommender.database import db_manager

        empty_collection = MagicMock()
        empty_collection.find.return_value = async_cursor([])

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=empty_collection)
        db_manager.db = mock_db

        results = await db_manager.search_repositories({"language": "Haskell"})

        assert results == []

    async def test_search_repositories_converts_object_id_to_string(self):
        from src.recommender.database import db_manager

        object_id = MagicMock()
        object_id.__str__ = lambda self: "507f1f77bcf86cd799439011"
        doc = {"_id": object_id, "name": "typed-repo"}

        mock_collection = MagicMock()
        mock_collection.find.return_value = async_cursor([doc])

        def _getitem(name):
            from src.recommender.config import settings

            if name == settings.repos_collection:
                return mock_collection
            empty = MagicMock()
            empty.find.return_value = async_cursor([])
            return empty

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(side_effect=_getitem)
        db_manager.db = mock_db

        results = await db_manager.search_repositories({})

        assert isinstance(results[0]["_id"], str)


# ---------------------------------------------------------------------------
# Tests: get_repositories_by_repo_ids
# ---------------------------------------------------------------------------


class TestGetRepositoriesByRepoIds:
    async def test_get_repositories_by_repo_ids_returns_empty_for_empty_input(self):
        from src.recommender.database import db_manager

        result = await db_manager.get_repositories_by_repo_ids([])

        assert result == {}

    async def test_get_repositories_by_repo_ids_builds_or_query(self):
        from src.recommender.database import db_manager

        doc = {"_id": "id1", "repo_id": "r1", "name": "repo-one"}

        mock_collection = MagicMock()
        mock_collection.find.return_value = async_cursor([doc])

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        db_manager.db = mock_db

        result = await db_manager.get_repositories_by_repo_ids(["r1"])

        mock_collection.find.assert_called_once()
        filter_arg = mock_collection.find.call_args[0][0]
        assert "$or" in filter_arg
        assert "r1" in result

    async def test_get_repositories_by_repo_ids_handles_exception_gracefully(self):
        from src.recommender.database import db_manager

        mock_collection = MagicMock()
        mock_collection.find.side_effect = Exception("DB connection lost")

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        db_manager.db = mock_db

        result = await db_manager.get_repositories_by_repo_ids(["r1", "r2"])

        assert result == {}


# ---------------------------------------------------------------------------
# Tests: vector_search
# ---------------------------------------------------------------------------


class TestVectorSearch:
    async def test_vector_search_awaits_qdrant_via_executor(self, mocker):
        from src.recommender.database import db_manager

        mock_loop = MagicMock()
        mock_loop.run_in_executor = AsyncMock(return_value=[])
        mocker.patch("src.recommender.database.asyncio.get_running_loop", return_value=mock_loop)

        db_manager.qdrant_client = MagicMock()

        await db_manager.vector_search([0.1, 0.2, 0.3], top_k=5)

        mock_loop.run_in_executor.assert_called_once()
        # First arg is executor (None), second is the lambda
        args = mock_loop.run_in_executor.call_args[0]
        assert args[0] is None
        assert callable(args[1])

    async def test_vector_search_maps_hits_to_dicts(self, mocker):
        from src.recommender.database import db_manager

        hit = MagicMock()
        hit.id = "repo_42"
        hit.score = 0.95
        hit.payload = {"name": "awesome-lib"}

        mock_loop = MagicMock()
        mock_loop.run_in_executor = AsyncMock(return_value=[hit])
        mocker.patch("src.recommender.database.asyncio.get_running_loop", return_value=mock_loop)

        db_manager.qdrant_client = MagicMock()

        results = await db_manager.vector_search([0.1, 0.2])

        assert len(results) == 1
        assert results[0]["repo_id"] == "repo_42"
        assert results[0]["score"] == 0.95
        assert results[0]["payload"] == {"name": "awesome-lib"}


# ---------------------------------------------------------------------------
# Tests: cache_get / cache_set
# ---------------------------------------------------------------------------


class TestCacheGet:
    async def test_cache_get_returns_none_when_cache_disabled(self, mocker):
        from src.recommender.database import db_manager

        mocker.patch("src.recommender.database.settings.enable_cache", False)

        db_manager.redis_client = AsyncMock()

        result = await db_manager.cache_get("some-key")

        assert result is None
        db_manager.redis_client.get.assert_not_called()

    async def test_cache_get_returns_parsed_json(self, mocker):
        from src.recommender.database import db_manager

        mocker.patch("src.recommender.database.settings.enable_cache", True)

        payload = {"results": [1, 2, 3]}
        mock_redis = AsyncMock()
        mock_redis.get.return_value = json.dumps(payload).encode()
        db_manager.redis_client = mock_redis

        result = await db_manager.cache_get("my-key")

        mock_redis.get.assert_called_once_with("reco:my-key")
        assert result == payload

    async def test_cache_get_returns_none_on_cache_miss(self, mocker):
        from src.recommender.database import db_manager

        mocker.patch("src.recommender.database.settings.enable_cache", True)

        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        db_manager.redis_client = mock_redis

        result = await db_manager.cache_get("missing-key")

        assert result is None


class TestCacheSet:
    async def test_cache_set_skips_when_cache_disabled(self, mocker):
        from src.recommender.database import db_manager

        mocker.patch("src.recommender.database.settings.enable_cache", False)

        mock_redis = AsyncMock()
        db_manager.redis_client = mock_redis

        await db_manager.cache_set("k", {"x": 1})

        mock_redis.setex.assert_not_called()

    async def test_cache_set_stores_with_ttl(self, mocker):
        from src.recommender.database import db_manager

        mocker.patch("src.recommender.database.settings.enable_cache", True)
        mocker.patch("src.recommender.database.settings.cache_ttl_seconds", 3600)

        mock_redis = AsyncMock()
        db_manager.redis_client = mock_redis

        await db_manager.cache_set("user:123", {"score": 0.9}, ttl=60)

        mock_redis.setex.assert_called_once()
        args = mock_redis.setex.call_args[0]
        assert args[0] == "reco:user:123"
        assert args[1] == 60
        # Third arg is the JSON-serialised value
        assert json.loads(args[2]) == {"score": 0.9}


# ---------------------------------------------------------------------------
# Tests: save_metrics / get_latest_metrics
# ---------------------------------------------------------------------------


class TestSaveMetrics:
    async def test_save_metrics_inserts_to_collection(self):
        from src.recommender.database import db_manager
        from src.recommender.models import EvaluationMetrics

        mock_collection = AsyncMock()
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        db_manager.db = mock_db

        now = datetime.now(timezone.utc)
        metrics = EvaluationMetrics(
            variant="baseline",
            mrr=0.42,
            click_through_rate=0.1,
            avg_response_time_ms=200.0,
            total_queries=100,
            total_interactions=15,
            evaluation_period_start=now - timedelta(days=7),
            evaluation_period_end=now,
        )

        await db_manager.save_metrics(metrics)

        mock_collection.insert_one.assert_called_once()
        inserted = mock_collection.insert_one.call_args[0][0]
        assert inserted["variant"] == "baseline"


class TestGetLatestMetrics:
    async def test_get_latest_metrics_returns_metrics_when_found(self):
        from src.recommender.database import db_manager

        now = datetime.now(timezone.utc)
        doc = {
            "variant": "hybrid",
            "mrr": 0.55,
            "click_through_rate": 0.2,
            "avg_response_time_ms": 150.0,
            "total_queries": 200,
            "total_interactions": 40,
            "evaluation_period_start": now - timedelta(days=7),
            "evaluation_period_end": now,
            "timestamp": now,
            "precision_at_k": {},
            "recall_at_k": {},
            "ndcg_at_k": {},
        }

        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = doc
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        db_manager.db = mock_db

        result = await db_manager.get_latest_metrics("hybrid")

        assert result is not None
        assert result.variant == "hybrid"
        assert result.mrr == 0.55

    async def test_get_latest_metrics_returns_none_when_not_found(self):
        from src.recommender.database import db_manager

        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = None
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        db_manager.db = mock_db

        result = await db_manager.get_latest_metrics("nonexistent")

        assert result is None


# ---------------------------------------------------------------------------
# Tests: get_active_ab_test
# ---------------------------------------------------------------------------


class TestGetActiveAbTest:
    async def test_get_active_ab_test_uses_timezone_aware_now(self, mocker):
        from src.recommender.database import db_manager

        fixed_now = datetime(2025, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_dt = mocker.patch("src.recommender.database.datetime")
        mock_dt.now.return_value = fixed_now

        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = None
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        db_manager.db = mock_db

        await db_manager.get_active_ab_test()

        mock_dt.now.assert_called_once_with(timezone.utc)
        query_arg = mock_collection.find_one.call_args[0][0]
        assert query_arg["start_date"]["$lte"] is fixed_now

    async def test_get_active_ab_test_returns_none_when_not_found(self):
        from src.recommender.database import db_manager

        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = None
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        db_manager.db = mock_db

        result = await db_manager.get_active_ab_test()

        assert result is None

    async def test_get_active_ab_test_strips_id_field(self):
        from src.recommender.database import db_manager

        now = datetime.now(timezone.utc)
        doc = {
            "_id": "mongo_id",
            "test_id": "t1",
            "name": "Experiment A",
            "description": "Testing hybrid vs baseline",
            "variants": ["baseline", "hybrid"],
            "traffic_split": {"baseline": 0.5, "hybrid": 0.5},
            "start_date": now - timedelta(days=1),
            "end_date": now + timedelta(days=7),
            "is_active": True,
            "created_at": now,
        }

        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = doc
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        db_manager.db = mock_db

        result = await db_manager.get_active_ab_test()

        assert result is not None
        assert result.test_id == "t1"
        # ABTestConfig has no _id field — successful construction proves stripping worked
        assert not hasattr(result, "_id")


# ---------------------------------------------------------------------------
# Tests: get_model_by_id
# ---------------------------------------------------------------------------


class TestGetModelById:
    async def test_get_model_by_id_returns_model_when_found(self):
        from src.recommender.database import db_manager

        now = datetime.now(timezone.utc)
        doc = {
            "model_id": "m-001",
            "model_type": "embedding",
            "variant": "v1",
            "version": "1.0.0",
            "path": "models/emb/v1",
            "is_active": True,
            "status": "active",
            "trained_at": now,
        }

        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = doc
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        db_manager.db = mock_db

        result = await db_manager.get_model_by_id("m-001")

        assert result is not None
        assert result.model_id == "m-001"
        mock_collection.find_one.assert_called_once_with({"model_id": "m-001"})

    async def test_get_model_by_id_returns_none_when_not_found(self):
        from src.recommender.database import db_manager

        mock_collection = AsyncMock()
        mock_collection.find_one.return_value = None
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        db_manager.db = mock_db

        result = await db_manager.get_model_by_id("nonexistent-id")

        assert result is None


# ---------------------------------------------------------------------------
# Tests: deactivate_models
# ---------------------------------------------------------------------------


class TestDeactivateModels:
    async def test_deactivate_models_calls_update_many_with_correct_filter(self):
        from src.recommender.database import db_manager

        mock_collection = AsyncMock()
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        db_manager.db = mock_db

        await db_manager.deactivate_models(model_type="embedding", variant="v1")

        mock_collection.update_many.assert_called_once_with(
            {"model_type": "embedding", "variant": "v1", "is_active": True},
            {"$set": {"is_active": False, "status": "archived"}},
        )


# ---------------------------------------------------------------------------
# Tests: activate_model
# ---------------------------------------------------------------------------


class TestActivateModel:
    async def test_activate_model_calls_update_one_with_correct_filter(self):
        from src.recommender.database import db_manager

        mock_collection = AsyncMock()
        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        db_manager.db = mock_db

        await db_manager.activate_model("m-007")

        mock_collection.update_one.assert_called_once_with(
            {"model_id": "m-007"},
            {"$set": {"is_active": True, "status": "active"}},
        )


# ---------------------------------------------------------------------------
# Tests: list_models_query
# ---------------------------------------------------------------------------


class TestListModelsQuery:
    def _make_model_doc(self, model_id: str, model_type: str, status: str) -> dict:
        return {
            "model_id": model_id,
            "model_type": model_type,
            "variant": "v1",
            "version": "1.0.0",
            "path": f"models/{model_id}",
            "is_active": status == "active",
            "status": status,
            "trained_at": datetime.now(timezone.utc),
        }

    async def test_list_models_query_no_filters(self):
        from src.recommender.database import db_manager

        docs = [
            self._make_model_doc("m1", "embedding", "active"),
            self._make_model_doc("m2", "cross_encoder", "candidate"),
        ]

        mock_collection = MagicMock()
        mock_collection.find.return_value = async_cursor(docs)

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        db_manager.db = mock_db

        results = await db_manager.list_models_query()

        mock_collection.find.assert_called_once_with({})
        assert len(results) == 2

    async def test_list_models_query_with_model_type_filter(self):
        from src.recommender.database import db_manager

        docs = [self._make_model_doc("m1", "embedding", "active")]

        mock_collection = MagicMock()
        mock_collection.find.return_value = async_cursor(docs)

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        db_manager.db = mock_db

        results = await db_manager.list_models_query(model_type="embedding")

        filter_arg = mock_collection.find.call_args[0][0]
        assert filter_arg == {"model_type": "embedding"}
        assert len(results) == 1
        assert results[0].model_type == "embedding"

    async def test_list_models_query_with_status_filter(self):
        from src.recommender.database import db_manager

        docs = [self._make_model_doc("m3", "personalization", "archived")]

        mock_collection = MagicMock()
        mock_collection.find.return_value = async_cursor(docs)

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        db_manager.db = mock_db

        results = await db_manager.list_models_query(status="archived")

        filter_arg = mock_collection.find.call_args[0][0]
        assert filter_arg == {"status": "archived"}
        assert results[0].status == "archived"

    async def test_list_models_query_with_both_filters(self):
        from src.recommender.database import db_manager

        docs = [self._make_model_doc("m4", "cross_encoder", "active")]

        mock_collection = MagicMock()
        mock_collection.find.return_value = async_cursor(docs)

        mock_db = MagicMock()
        mock_db.__getitem__ = MagicMock(return_value=mock_collection)
        db_manager.db = mock_db

        results = await db_manager.list_models_query(model_type="cross_encoder", status="active")

        filter_arg = mock_collection.find.call_args[0][0]
        assert filter_arg == {"model_type": "cross_encoder", "status": "active"}
        assert len(results) == 1
