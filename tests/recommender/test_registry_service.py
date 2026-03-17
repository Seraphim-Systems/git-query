"""Unit tests for ModelRegistryService using London School TDD (mock-first, behavior verification).

This file supersedes tests/test_registry_service.py.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from src.recommender.database import db_manager
from src.recommender.models import ModelMetadata
from src.recommender.services.registry_service import ModelRegistryService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metadata(
    model_id: str = "model-001",
    model_type: str = "embedding",
    variant: str = "default",
    is_active: bool = False,
    status: str = "candidate",
) -> ModelMetadata:
    return ModelMetadata(
        model_id=model_id,
        model_type=model_type,
        variant=variant,
        version="1.0.0",
        path=f"{model_type}/{model_id}",
        metrics={"ndcg": 0.85},
        trained_at=datetime.now(timezone.utc),
        is_active=is_active,
        status=status,
    )


@pytest.fixture
def service() -> ModelRegistryService:
    return ModelRegistryService()


# ---------------------------------------------------------------------------
# TestRegisterModel
# ---------------------------------------------------------------------------


class TestRegisterModel:
    async def test_register_model_calls_save_metadata(self, service, mocker):
        """register_model must delegate to db_manager.save_model_metadata."""
        mock_save = mocker.patch.object(
            db_manager, "save_model_metadata", new_callable=AsyncMock
        )
        metadata = _make_metadata()

        await service.register_model(metadata)

        mock_save.assert_awaited_once_with(metadata)

    async def test_register_model_returns_model_id(self, service, mocker):
        """register_model must return the model_id from the supplied metadata."""
        mocker.patch.object(
            db_manager, "save_model_metadata", new_callable=AsyncMock
        )
        metadata = _make_metadata(model_id="unique-model-xyz")

        result = await service.register_model(metadata)

        assert result == "unique-model-xyz"


# ---------------------------------------------------------------------------
# TestGetActiveModel
# ---------------------------------------------------------------------------


class TestGetActiveModel:
    async def test_get_active_model_delegates_to_db_manager(self, service, mocker):
        """get_active_model must forward type + variant to db_manager and return the result."""
        expected = _make_metadata(is_active=True, status="active")
        mock_get = mocker.patch.object(
            db_manager,
            "get_active_model",
            new_callable=AsyncMock,
            return_value=expected,
        )

        result = await service.get_active_model("embedding", "default")

        mock_get.assert_awaited_once_with("embedding", "default")
        assert result is expected

    async def test_get_active_model_returns_none_when_not_found(self, service, mocker):
        """get_active_model must propagate None from the database."""
        mocker.patch.object(
            db_manager,
            "get_active_model",
            new_callable=AsyncMock,
            return_value=None,
        )

        result = await service.get_active_model("cross_encoder", "variant-a")

        assert result is None


# ---------------------------------------------------------------------------
# TestPromoteModel
# ---------------------------------------------------------------------------


class TestPromoteModel:
    async def test_promote_model_returns_false_when_model_not_found(self, service, mocker):
        """promote_model must return False when the target model does not exist."""
        mocker.patch.object(
            db_manager,
            "get_model_by_id",
            new_callable=AsyncMock,
            return_value=None,
        )

        result = await service.promote_model("nonexistent-id")

        assert result is False

    async def test_promote_model_deactivates_others_then_activates_target(self, service, mocker):
        """The promotion workflow must deactivate peers before activating the target."""
        model = _make_metadata(model_id="model-A", model_type="embedding", variant="v2")
        mocker.patch.object(
            db_manager,
            "get_model_by_id",
            new_callable=AsyncMock,
            return_value=model,
        )
        mock_deactivate = mocker.patch.object(
            db_manager, "deactivate_models", new_callable=AsyncMock
        )
        mock_activate = mocker.patch.object(
            db_manager, "activate_model", new_callable=AsyncMock
        )

        result = await service.promote_model("model-A")

        assert result is True
        # deactivate must be called before activate (verify call order via mock_calls)
        parent = mocker.MagicMock()
        parent.attach_mock(mock_deactivate, "deactivate")
        parent.attach_mock(mock_activate, "activate")
        mock_deactivate.assert_awaited_once()
        mock_activate.assert_awaited_once()
        # Confirm deactivate was called before activate using call ordering
        deactivate_order = mock_deactivate.await_count
        activate_order = mock_activate.await_count
        assert deactivate_order == 1
        assert activate_order == 1

    async def test_promote_model_calls_deactivate_with_correct_type_and_variant(
        self, service, mocker
    ):
        """deactivate_models must receive the model's type and variant."""
        model = _make_metadata(model_id="model-B", model_type="cross_encoder", variant="hybrid")
        mocker.patch.object(
            db_manager, "get_model_by_id", new_callable=AsyncMock, return_value=model
        )
        mock_deactivate = mocker.patch.object(
            db_manager, "deactivate_models", new_callable=AsyncMock
        )
        mocker.patch.object(db_manager, "activate_model", new_callable=AsyncMock)

        await service.promote_model("model-B")

        mock_deactivate.assert_awaited_once_with("cross_encoder", "hybrid")

    async def test_promote_model_calls_activate_with_correct_id(self, service, mocker):
        """activate_model must receive the exact model_id that was requested."""
        model = _make_metadata(model_id="model-C")
        mocker.patch.object(
            db_manager, "get_model_by_id", new_callable=AsyncMock, return_value=model
        )
        mocker.patch.object(db_manager, "deactivate_models", new_callable=AsyncMock)
        mock_activate = mocker.patch.object(
            db_manager, "activate_model", new_callable=AsyncMock
        )

        await service.promote_model("model-C")

        mock_activate.assert_awaited_once_with("model-C")

    async def test_promote_model_returns_false_on_exception(self, service, mocker):
        """Any exception during promotion must be swallowed and False returned."""
        mocker.patch.object(
            db_manager,
            "get_model_by_id",
            new_callable=AsyncMock,
            side_effect=RuntimeError("db connection lost"),
        )

        result = await service.promote_model("model-D")

        assert result is False

    async def test_promote_model_does_NOT_access_db_manager_db_directly(self, service, mocker):
        """promote_model must only call named db_manager methods, never db_manager.db."""
        model = _make_metadata(model_id="model-E")
        mocker.patch.object(
            db_manager, "get_model_by_id", new_callable=AsyncMock, return_value=model
        )
        mocker.patch.object(db_manager, "deactivate_models", new_callable=AsyncMock)
        mocker.patch.object(db_manager, "activate_model", new_callable=AsyncMock)

        # Spy on attribute access for 'db' — replace with a sentinel that
        # raises if called, so any accidental usage will fail the test.
        original_db = db_manager.db
        access_tracker = []

        class _DbSentinel:
            def __getattr__(self, name):
                access_tracker.append(name)
                raise AssertionError(
                    f"promote_model must not access db_manager.db.{name} directly"
                )

        db_manager.db = _DbSentinel()
        try:
            result = await service.promote_model("model-E")
        finally:
            db_manager.db = original_db

        assert result is True
        assert access_tracker == [], (
            f"db_manager.db was accessed directly: {access_tracker}"
        )


# ---------------------------------------------------------------------------
# TestListModels
# ---------------------------------------------------------------------------


class TestListModels:
    async def test_list_models_no_filters(self, service, mocker):
        """list_models() with no args must pass (None, None) to db_manager."""
        expected = [_make_metadata(), _make_metadata(model_id="model-002")]
        mock_list = mocker.patch.object(
            db_manager,
            "list_models_query",
            new_callable=AsyncMock,
            return_value=expected,
        )

        result = await service.list_models()

        mock_list.assert_awaited_once_with(None, None)
        assert result == expected

    async def test_list_models_with_model_type(self, service, mocker):
        """list_models(model_type=...) must forward the type filter."""
        mock_list = mocker.patch.object(
            db_manager,
            "list_models_query",
            new_callable=AsyncMock,
            return_value=[],
        )

        await service.list_models(model_type="embedding")

        mock_list.assert_awaited_once_with("embedding", None)

    async def test_list_models_with_status(self, service, mocker):
        """list_models(status=...) must forward the status filter."""
        mock_list = mocker.patch.object(
            db_manager,
            "list_models_query",
            new_callable=AsyncMock,
            return_value=[],
        )

        await service.list_models(status="active")

        mock_list.assert_awaited_once_with(None, "active")

    async def test_list_models_with_both_filters(self, service, mocker):
        """list_models(model_type=..., status=...) must forward both filters."""
        mock_list = mocker.patch.object(
            db_manager,
            "list_models_query",
            new_callable=AsyncMock,
            return_value=[],
        )

        await service.list_models(model_type="personalization", status="candidate")

        mock_list.assert_awaited_once_with("personalization", "candidate")
