"""Shared fixtures for the recommender test suite.

All external dependencies (MongoDB, Qdrant, Redis, ML models) are mocked so
these tests run entirely in-process with no network or disk I/O.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import pytest_asyncio

from src.recommender.models import (
    ABTestConfig,
    ModelMetadata,
    RecommendationRequest,
    RepositoryResult,
    UserPreferences,
)
from src.recommender.services.embedding_service import EmbeddingService
from src.recommender.services.reranker_service import RerankerService
from src.recommender.engines.baseline import BaselineEngine
from src.recommender.engines.hybrid import HybridRetrievalEngine
from src.recommender.engines.personalized import PersonalizedEngine


# ---------------------------------------------------------------------------
# A/B testing guard
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def disable_ab_testing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Disable A/B testing globally so variant selection is deterministic."""
    monkeypatch.setattr(
        "src.recommender.config.settings.ab_test_enabled",
        False,
        raising=False,
    )
    monkeypatch.setattr(
        "src.recommender.config.settings.default_variant",
        "baseline",
        raising=False,
    )


# ---------------------------------------------------------------------------
# Database mock
# ---------------------------------------------------------------------------


class _AsyncCursor:
    """Minimal async cursor that supports __aiter__, .sort(), .limit(), .skip()."""

    def __init__(self, docs: list[dict[str, Any]]) -> None:
        self._docs = docs
        self._index = 0

    # Chainable stubs — each returns self so callers can do .sort().limit()
    def sort(self, *_args: Any, **_kwargs: Any) -> "_AsyncCursor":
        return self

    def limit(self, *_args: Any, **_kwargs: Any) -> "_AsyncCursor":
        return self

    def skip(self, *_args: Any, **_kwargs: Any) -> "_AsyncCursor":
        return self

    def __aiter__(self) -> "_AsyncCursor":
        self._index = 0
        return self

    async def __anext__(self) -> dict[str, Any]:
        if self._index >= len(self._docs):
            raise StopAsyncIteration
        doc = self._docs[self._index]
        self._index += 1
        return doc


@pytest.fixture
def mock_db(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """
    Inject a fully-mocked DatabaseManager into the module-level ``db_manager``
    singleton used by all engine and service code.

    The fixture exposes a helper ``_async_cursor(docs)`` factory on the mock
    so individual tests can configure collection stubs without reimplementing
    the cursor protocol themselves.
    """
    db = MagicMock()

    # Helper that tests can call to build a properly behaving async cursor.
    db._async_cursor = _AsyncCursor

    # --- MongoDB collection proxy ---
    collection = MagicMock()
    collection.find.return_value = _AsyncCursor([])
    collection.find_one = AsyncMock(return_value=None)
    collection.insert_one = AsyncMock(return_value=MagicMock(inserted_id="mock-id"))
    collection.update_one = AsyncMock(return_value=None)
    collection.update_many = AsyncMock(return_value=None)
    collection.create_index = AsyncMock(return_value=None)

    db_mock = MagicMock()
    db_mock.__getitem__ = MagicMock(return_value=collection)
    db.db = db_mock

    # --- High-level async methods ---
    db.connect = AsyncMock()
    db.close = AsyncMock()
    db.log_interaction = AsyncMock(return_value="mock-interaction-id")
    db.get_user_interactions = AsyncMock(return_value=[])
    db.get_user_preferences = AsyncMock(return_value=None)
    db.update_user_preferences = AsyncMock(return_value=None)
    db.search_repositories = AsyncMock(return_value=[])
    db.get_repositories_by_repo_ids = AsyncMock(return_value={})
    db.vector_search = AsyncMock(return_value=[])
    db.cache_get = AsyncMock(return_value=None)
    db.cache_set = AsyncMock(return_value=None)
    db.save_metrics = AsyncMock(return_value=None)
    db.get_latest_metrics = AsyncMock(return_value=None)
    db.get_active_ab_test = AsyncMock(return_value=None)
    db.save_model_metadata = AsyncMock(return_value=None)
    db.get_active_model = AsyncMock(return_value=None)
    db.get_model_by_id = AsyncMock(return_value=None)
    db.deactivate_models = AsyncMock(return_value=None)
    db.activate_model = AsyncMock(return_value=None)
    db.list_models_query = AsyncMock(return_value=[])

    # Patch the singleton used by all engine/service modules
    monkeypatch.setattr("src.recommender.database.db_manager", db)
    monkeypatch.setattr("src.recommender.engines.baseline.db_manager", db)
    monkeypatch.setattr("src.recommender.engines.hybrid.db_manager", db)
    monkeypatch.setattr("src.recommender.engines.personalized.db_manager", db)

    return db


# ---------------------------------------------------------------------------
# ML model mocks
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_sentence_transformer() -> MagicMock:
    """SentenceTransformer stub — no disk I/O, deterministic zero vector."""
    model = MagicMock()
    model.encode.return_value = np.zeros(384, dtype=np.float32)
    model.get_sentence_embedding_dimension.return_value = 384
    return model


@pytest.fixture
def mock_cross_encoder() -> MagicMock:
    """CrossEncoder stub — returns fixed scores for up to four candidates."""
    model = MagicMock()
    model.predict.return_value = np.array([0.9, 0.7, 0.4, 0.2], dtype=np.float32)
    return model


# ---------------------------------------------------------------------------
# Service fixtures (pre-loaded, no disk I/O)
# ---------------------------------------------------------------------------


@pytest.fixture
def embedding_service(mock_sentence_transformer: MagicMock) -> EmbeddingService:
    """EmbeddingService with the mock transformer already in place."""
    svc = EmbeddingService.__new__(EmbeddingService)
    svc.model_name = "fake-model"
    svc.model = mock_sentence_transformer
    svc.current_model_id = None
    svc._loaded_path = "fake-model"
    svc.device = "cpu"
    return svc


@pytest.fixture
def reranker_service(mock_cross_encoder: MagicMock) -> RerankerService:
    """RerankerService with the mock cross-encoder already in place."""
    svc = RerankerService.__new__(RerankerService)
    svc.model_name = "fake-model"
    svc.model = mock_cross_encoder
    svc.current_model_id = None
    svc._loaded_path = "fake-model"
    return svc


# ---------------------------------------------------------------------------
# Domain object factories
# ---------------------------------------------------------------------------


@pytest.fixture
def make_repo_result():
    """Factory for RepositoryResult with sensible defaults; accepts **kwargs overrides."""

    def _factory(**kwargs: Any) -> RepositoryResult:
        defaults: dict[str, Any] = {
            "repo_id": "owner/repo",
            "name": "repo",
            "full_name": "owner/repo",
            "description": "A sample repository",
            "language": "Python",
            "stars": 1000,
            "forks": 100,
            "url": "https://github.com/owner/repo",
            "license": "MIT",
            "last_updated": datetime.now(timezone.utc),
            "score": 0.85,
            "rank": 1,
            "explanation": {"method": "test"},
        }
        defaults.update(kwargs)
        return RepositoryResult(**defaults)

    return _factory


@pytest.fixture
def make_request():
    """Factory for RecommendationRequest with sensible defaults; accepts **kwargs overrides."""

    def _factory(**kwargs: Any) -> RecommendationRequest:
        defaults: dict[str, Any] = {
            "query": "python web framework",
            "top_k": 10,
        }
        defaults.update(kwargs)
        return RecommendationRequest(**defaults)

    return _factory


# ---------------------------------------------------------------------------
# Sample domain objects
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_user_prefs() -> UserPreferences:
    return UserPreferences(
        user_id="user-123",
        language_preferences={"Python": 0.9, "Go": 0.3},
        topic_preferences={},
        total_interactions=10,
        last_updated=datetime.now(timezone.utc),
    )


@pytest.fixture
def sample_ab_test() -> ABTestConfig:
    return ABTestConfig(
        test_id="test-abc",
        name="Hybrid vs Baseline",
        description="Compare hybrid retrieval against keyword-only baseline",
        variants=["baseline", "hybrid"],
        traffic_split={"baseline": 0.5, "hybrid": 0.5},
        start_date=datetime.now(timezone.utc),
        end_date=None,
        is_active=True,
    )


@pytest.fixture
def sample_model_metadata() -> ModelMetadata:
    return ModelMetadata(
        model_id="emb-model-001",
        model_type="embedding",
        variant="default",
        version="1.0.0",
        path="embedding/emb-model-001",
        hyperparameters={},
        metrics={},
        trained_at=datetime.now(timezone.utc),
        is_active=True,
        status="active",
    )


# ---------------------------------------------------------------------------
# Engine fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def baseline_engine(mock_db: MagicMock) -> BaselineEngine:  # noqa: ARG001
    """BaselineEngine with db_manager patched out."""
    return BaselineEngine()


@pytest.fixture
def hybrid_engine(
    mock_db: MagicMock,  # noqa: ARG001
    embedding_service: EmbeddingService,
    reranker_service: RerankerService,
) -> HybridRetrievalEngine:
    """HybridRetrievalEngine wired to mock services and mock db."""
    engine = HybridRetrievalEngine(
        embedding_service=embedding_service,
        reranker_service=reranker_service,
    )
    return engine


@pytest.fixture
def personalized_engine(
    mock_db: MagicMock,  # noqa: ARG001
    embedding_service: EmbeddingService,
    reranker_service: RerankerService,
) -> PersonalizedEngine:
    """PersonalizedEngine wired to mock services and mock db."""
    engine = PersonalizedEngine(
        embedding_service=embedding_service,
        reranker_service=reranker_service,
    )
    return engine
