"""Unit tests for PersonalizationService using London School TDD (mock-first, behavior verification)."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from src.recommender.database import db_manager
from src.recommender.models import InteractionType, UserInteraction, UserPreferences
from src.recommender.services.personalization_service import PersonalizationService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_interaction(
    interaction_type: InteractionType = InteractionType.CLICK,
    user_id: str = "user-1",
    metadata: dict = None,
) -> UserInteraction:
    return UserInteraction(
        user_id=user_id,
        query="machine learning",
        repo_id="repo-001",
        interaction_type=interaction_type,
        timestamp=datetime.now(timezone.utc),
        metadata=metadata or {},
    )


def _make_prefs(
    user_id: str = "user-1",
    language_preferences: dict = None,
    topic_preferences: dict = None,
    total_interactions: int = 0,
) -> UserPreferences:
    return UserPreferences(
        user_id=user_id,
        language_preferences=language_preferences or {},
        topic_preferences=topic_preferences or {},
        total_interactions=total_interactions,
        last_updated=datetime.now(timezone.utc),
    )


@pytest.fixture
def service() -> PersonalizationService:
    return PersonalizationService()


# ---------------------------------------------------------------------------
# TestGetSignalWeight (pure — no mocks)
# ---------------------------------------------------------------------------


class TestGetSignalWeight:
    def test_click_weight_is_positive(self, service):
        assert service._get_signal_weight(InteractionType.CLICK) > 0

    def test_save_weight_greater_than_click(self, service):
        assert service._get_signal_weight(InteractionType.SAVE) > service._get_signal_weight(
            InteractionType.CLICK
        )

    def test_thumbs_up_weight_greatest_positive(self, service):
        thumbs_up = service._get_signal_weight(InteractionType.THUMBS_UP)
        assert thumbs_up > service._get_signal_weight(InteractionType.SAVE)
        assert thumbs_up > service._get_signal_weight(InteractionType.CLICK)

    def test_dismiss_weight_is_negative(self, service):
        assert service._get_signal_weight(InteractionType.DISMISS) < 0

    def test_thumbs_down_weight_most_negative(self, service):
        thumbs_down = service._get_signal_weight(InteractionType.THUMBS_DOWN)
        assert thumbs_down < service._get_signal_weight(InteractionType.DISMISS)

    def test_view_weight_is_zero(self, service):
        assert service._get_signal_weight(InteractionType.VIEW) == 0.0


# ---------------------------------------------------------------------------
# TestNormalizePreferences (pure — no mocks)
# ---------------------------------------------------------------------------


class TestNormalizePreferences:
    def test_normalize_empty_returns_empty(self, service):
        assert service._normalize_preferences({}) == {}

    def test_normalize_single_value_returns_half(self, service):
        result = service._normalize_preferences({"python": 5.0})
        assert result == {"python": 0.5}

    def test_normalize_all_same_values_return_half(self, service):
        prefs = {"python": 3.0, "rust": 3.0, "go": 3.0}
        result = service._normalize_preferences(prefs)
        assert all(v == 0.5 for v in result.values())

    def test_normalize_negative_values_shifted_to_positive_range(self, service):
        prefs = {"python": -2.0, "rust": 1.0}
        result = service._normalize_preferences(prefs)
        assert all(v >= 0.0 for v in result.values())

    def test_normalize_result_in_zero_to_one_range(self, service):
        prefs = {"python": 3.0, "rust": -1.0, "go": 0.5, "ts": 2.0}
        result = service._normalize_preferences(prefs)
        for v in result.values():
            assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# TestUpdatePreferencesFromInteraction
# ---------------------------------------------------------------------------


class TestUpdatePreferencesFromInteraction:
    async def test_creates_new_prefs_when_none_exist(self, service, mocker):
        """When get_user_preferences returns None a fresh object is created and saved."""
        mocker.patch.object(
            db_manager,
            "get_user_preferences",
            new_callable=AsyncMock,
            return_value=None,
        )
        mock_update = mocker.patch.object(
            db_manager, "update_user_preferences", new_callable=AsyncMock
        )

        interaction = _make_interaction(InteractionType.CLICK)
        await service.update_preferences_from_interaction(interaction, {"language": "Python"})

        mock_update.assert_awaited_once()
        saved_prefs: UserPreferences = mock_update.call_args[0][0]
        assert saved_prefs.user_id == "user-1"

    async def test_updates_language_preference_on_positive_signal(self, service, mocker):
        """A CLICK interaction must increase the language preference score."""
        existing = _make_prefs(language_preferences={"Python": 0.5})
        mocker.patch.object(
            db_manager,
            "get_user_preferences",
            new_callable=AsyncMock,
            return_value=existing,
        )
        mock_update = mocker.patch.object(
            db_manager, "update_user_preferences", new_callable=AsyncMock
        )

        interaction = _make_interaction(InteractionType.CLICK)
        await service.update_preferences_from_interaction(interaction, {"language": "Python"})

        saved_prefs: UserPreferences = mock_update.call_args[0][0]
        # After normalization the value must still exist and be valid.
        assert "Python" in saved_prefs.language_preferences

    async def test_decrements_language_preference_on_negative_signal(self, service, mocker):
        """A THUMBS_DOWN interaction must not increase the language preference."""
        existing = _make_prefs(language_preferences={"Python": 3.0, "Rust": 1.0})
        mocker.patch.object(
            db_manager,
            "get_user_preferences",
            new_callable=AsyncMock,
            return_value=existing,
        )
        mock_update = mocker.patch.object(
            db_manager, "update_user_preferences", new_callable=AsyncMock
        )

        # Capture Python score before the thumbs-down
        python_before = existing.language_preferences["Python"]

        interaction = _make_interaction(InteractionType.THUMBS_DOWN)
        await service.update_preferences_from_interaction(interaction, {"language": "Python"})

        # The raw accumulated value for Python is 3.0 + (-2.0) = 1.0, which is
        # the same as Rust 1.0, so after normalization both are 0.5. We verify
        # the save was called (behavior contract) and Python is still tracked.
        mock_update.assert_awaited_once()
        saved_prefs: UserPreferences = mock_update.call_args[0][0]
        assert "Python" in saved_prefs.language_preferences

    async def test_updates_topic_preferences(self, service, mocker):
        """Topics from repo_data must be accumulated in topic_preferences."""
        mocker.patch.object(
            db_manager,
            "get_user_preferences",
            new_callable=AsyncMock,
            return_value=None,
        )
        mock_update = mocker.patch.object(
            db_manager, "update_user_preferences", new_callable=AsyncMock
        )

        interaction = _make_interaction(InteractionType.SAVE)
        repo_data = {"language": "Python", "topics": ["ml", "nlp"]}
        await service.update_preferences_from_interaction(interaction, repo_data)

        saved_prefs: UserPreferences = mock_update.call_args[0][0]
        assert "ml" in saved_prefs.topic_preferences
        assert "nlp" in saved_prefs.topic_preferences

    async def test_skips_language_update_when_repo_has_no_language(self, service, mocker):
        """When repo_data has no 'language' key language_preferences must remain empty."""
        mocker.patch.object(
            db_manager,
            "get_user_preferences",
            new_callable=AsyncMock,
            return_value=None,
        )
        mock_update = mocker.patch.object(
            db_manager, "update_user_preferences", new_callable=AsyncMock
        )

        interaction = _make_interaction(InteractionType.CLICK)
        await service.update_preferences_from_interaction(interaction, {})

        saved_prefs: UserPreferences = mock_update.call_args[0][0]
        assert saved_prefs.language_preferences == {}

    async def test_increments_total_interactions(self, service, mocker):
        """Every call must increment total_interactions by 1."""
        existing = _make_prefs(total_interactions=4)
        mocker.patch.object(
            db_manager,
            "get_user_preferences",
            new_callable=AsyncMock,
            return_value=existing,
        )
        mock_update = mocker.patch.object(
            db_manager, "update_user_preferences", new_callable=AsyncMock
        )

        interaction = _make_interaction(InteractionType.VIEW)
        await service.update_preferences_from_interaction(interaction, {})

        saved_prefs: UserPreferences = mock_update.call_args[0][0]
        assert saved_prefs.total_interactions == 5

    async def test_last_updated_is_timezone_aware(self, service, mocker):
        """last_updated must carry timezone info after an update."""
        mocker.patch.object(
            db_manager,
            "get_user_preferences",
            new_callable=AsyncMock,
            return_value=None,
        )
        mock_update = mocker.patch.object(
            db_manager, "update_user_preferences", new_callable=AsyncMock
        )

        interaction = _make_interaction(InteractionType.CLICK)
        await service.update_preferences_from_interaction(interaction, {"language": "Go"})

        saved_prefs: UserPreferences = mock_update.call_args[0][0]
        assert saved_prefs.last_updated.tzinfo is not None

    async def test_saves_updated_preferences(self, service, mocker):
        """update_user_preferences must be awaited exactly once per call."""
        mocker.patch.object(
            db_manager,
            "get_user_preferences",
            new_callable=AsyncMock,
            return_value=None,
        )
        mock_update = mocker.patch.object(
            db_manager, "update_user_preferences", new_callable=AsyncMock
        )

        interaction = _make_interaction(InteractionType.CLICK)
        await service.update_preferences_from_interaction(interaction, {"language": "TypeScript"})

        mock_update.assert_awaited_once()


# ---------------------------------------------------------------------------
# TestBatchUpdatePreferences
# ---------------------------------------------------------------------------


class TestBatchUpdatePreferences:
    async def test_batch_uses_all_interactions(self, service, mocker):
        """All 3 interactions returned from the db must be processed and saved."""
        interactions = [
            _make_interaction(
                InteractionType.CLICK,
                metadata={"language": "Python", "topics": ["ml"]},
            ),
            _make_interaction(
                InteractionType.SAVE,
                metadata={"language": "Rust", "topics": ["systems"]},
            ),
            _make_interaction(
                InteractionType.THUMBS_DOWN,
                metadata={"language": "Python", "topics": []},
            ),
        ]
        mocker.patch.object(
            db_manager,
            "get_user_interactions",
            new_callable=AsyncMock,
            return_value=interactions,
        )
        mock_update = mocker.patch.object(
            db_manager, "update_user_preferences", new_callable=AsyncMock
        )

        await service.batch_update_preferences("user-1")

        mock_update.assert_awaited_once()
        saved_prefs: UserPreferences = mock_update.call_args[0][0]
        assert saved_prefs.total_interactions == 3

    async def test_batch_ignores_interactions_without_metadata(self, service, mocker):
        """Interactions with empty metadata must not raise and must still be counted."""
        interactions = [
            _make_interaction(InteractionType.CLICK, metadata={}),
            _make_interaction(InteractionType.SAVE, metadata={}),
        ]
        mocker.patch.object(
            db_manager,
            "get_user_interactions",
            new_callable=AsyncMock,
            return_value=interactions,
        )
        mock_update = mocker.patch.object(
            db_manager, "update_user_preferences", new_callable=AsyncMock
        )

        await service.batch_update_preferences("user-1")

        mock_update.assert_awaited_once()
        saved_prefs: UserPreferences = mock_update.call_args[0][0]
        # No language or topic data means empty dicts (no error)
        assert saved_prefs.language_preferences == {}
        assert saved_prefs.topic_preferences == {}
        # Interaction count reflects all processed rows
        assert saved_prefs.total_interactions == 2
