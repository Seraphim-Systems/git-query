"""Unit tests for PersonalizedEngine (London School TDD — mock-first)."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest

from src.recommender.database import db_manager
from src.recommender.engines.personalized import PersonalizedEngine
from src.recommender.models import RecommendationRequest, RepositoryResult, UserPreferences


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_request(
    query: str = "python web framework",
    top_k: int = 5,
    user_id: str | None = "user-42",
    enable_personalization: bool = True,
    language: str | None = None,
) -> RecommendationRequest:
    return RecommendationRequest(
        query=query,
        top_k=top_k,
        user_id=user_id,
        enable_personalization=enable_personalization,
        language=language,
    )


def _make_prefs(
    user_id: str = "user-42",
    language_preferences: dict | None = None,
    total_interactions: int = 10,
) -> UserPreferences:
    return UserPreferences(
        user_id=user_id,
        language_preferences=language_preferences or {"Python": 0.8, "Rust": 0.3},
        total_interactions=total_interactions,
        last_updated=datetime.now(timezone.utc),
    )


def _make_result(
    repo_id: str = "owner/repo",
    language: str | None = "Python",
    score: float = 0.5,
    rank: int = 1,
) -> RepositoryResult:
    return RepositoryResult(
        repo_id=repo_id,
        name=repo_id.split("/")[-1],
        full_name=repo_id,
        description="Test repository",
        language=language,
        stars=100,
        forks=10,
        url=f"https://github.com/{repo_id}",
        license="MIT",
        last_updated=datetime.now(timezone.utc),
        score=score,
        rank=rank,
        explanation={"method": "hybrid_rrf", "sources": ["semantic"], "rrf_score": score},
    )


def _make_engine(embedding_service=None, reranker_service=None) -> PersonalizedEngine:
    return PersonalizedEngine(
        embedding_service=embedding_service,
        reranker_service=reranker_service,
    )


# ---------------------------------------------------------------------------
# TestApplyPersonalization
# ---------------------------------------------------------------------------


class TestApplyPersonalization:
    async def test_no_personalization_when_prefs_is_none(self, mocker):
        mocker.patch.object(
            db_manager, "get_user_preferences", new_callable=AsyncMock, return_value=None
        )
        engine = _make_engine()
        original = [_make_result("a/a", score=0.9, rank=1), _make_result("b/b", score=0.5, rank=2)]
        request = _make_request()

        results = await engine._apply_personalization(original, request)

        # Order and scores must be unchanged
        assert [r.repo_id for r in results] == ["a/a", "b/b"]
        assert results[0].score == pytest.approx(0.9)
        assert results[1].score == pytest.approx(0.5)

    async def test_no_personalization_below_interaction_threshold(self, mocker):
        """Engine returns results unchanged when total_interactions < min threshold."""
        from src.recommender.config import settings

        low_interaction_prefs = _make_prefs(
            total_interactions=settings.min_interactions_for_personalization - 1
        )
        mocker.patch.object(
            db_manager,
            "get_user_preferences",
            new_callable=AsyncMock,
            return_value=low_interaction_prefs,
        )
        engine = _make_engine()
        original = [_make_result("a/a", score=0.9, rank=1)]
        results = await engine._apply_personalization(original, _make_request())

        assert results[0].score == pytest.approx(0.9)

    async def test_boosts_preferred_language(self, mocker):
        """A result whose language is in prefs should have score > original score."""
        from src.recommender.config import settings

        prefs = _make_prefs(language_preferences={"Python": 0.8}, total_interactions=settings.min_interactions_for_personalization)
        mocker.patch.object(
            db_manager, "get_user_preferences", new_callable=AsyncMock, return_value=prefs
        )
        engine = _make_engine()
        original_score = 0.5
        result = _make_result("a/a", language="Python", score=original_score, rank=1)

        results = await engine._apply_personalization([result], _make_request())

        assert results[0].score > original_score

    async def test_boost_is_proportional_to_language_preference_score(self, mocker):
        """score = original * (1 + lang_pref * personalization_weight)."""
        from src.recommender.config import settings

        lang_pref = 0.6
        prefs = _make_prefs(
            language_preferences={"TypeScript": lang_pref},
            total_interactions=settings.min_interactions_for_personalization,
        )
        mocker.patch.object(
            db_manager, "get_user_preferences", new_callable=AsyncMock, return_value=prefs
        )
        engine = _make_engine()
        original_score = 0.4
        result = _make_result("a/a", language="TypeScript", score=original_score, rank=1)

        results = await engine._apply_personalization([result], _make_request())

        expected = original_score * (1 + lang_pref * settings.personalization_weight)
        assert results[0].score == pytest.approx(expected)

    async def test_no_boost_for_non_preferred_language(self, mocker):
        """A language absent from prefs gets no boost."""
        from src.recommender.config import settings

        prefs = _make_prefs(
            language_preferences={"Python": 0.9},
            total_interactions=settings.min_interactions_for_personalization,
        )
        mocker.patch.object(
            db_manager, "get_user_preferences", new_callable=AsyncMock, return_value=prefs
        )
        engine = _make_engine()
        original_score = 0.7
        result = _make_result("a/a", language="Haskell", score=original_score, rank=1)

        results = await engine._apply_personalization([result], _make_request())

        assert results[0].score == pytest.approx(original_score)

    async def test_results_re_sorted_after_boost(self, mocker):
        """A lower-ranked result with a preferred language should move above a higher-scored one."""
        from src.recommender.config import settings

        prefs = _make_prefs(
            language_preferences={"Rust": 1.0},
            total_interactions=settings.min_interactions_for_personalization,
        )
        mocker.patch.object(
            db_manager, "get_user_preferences", new_callable=AsyncMock, return_value=prefs
        )
        engine = _make_engine()
        # result_a has higher base score but non-preferred language
        result_a = _make_result("a/a", language="Go", score=0.8, rank=1)
        # result_b score chosen so that 0.75 * (1 + 1.0 * 0.15) = 0.8625 > 0.8
        result_b = _make_result("b/b", language="Rust", score=0.75, rank=2)

        results = await engine._apply_personalization([result_a, result_b], _make_request())

        assert results[0].repo_id == "b/b"

    async def test_ranks_reassigned_after_resort(self, mocker):
        """After re-sorting, rank values must be sequential starting from 1."""
        from src.recommender.config import settings

        prefs = _make_prefs(
            language_preferences={"Rust": 1.0},
            total_interactions=settings.min_interactions_for_personalization,
        )
        mocker.patch.object(
            db_manager, "get_user_preferences", new_callable=AsyncMock, return_value=prefs
        )
        engine = _make_engine()
        result_a = _make_result("a/a", language="Go", score=0.8, rank=1)
        result_b = _make_result("b/b", language="Rust", score=0.5, rank=2)
        result_c = _make_result("c/c", language="Go", score=0.3, rank=3)

        results = await engine._apply_personalization([result_a, result_b, result_c], _make_request())

        assert [r.rank for r in results] == [1, 2, 3]

    async def test_explanation_updated_with_boost_and_original_score(self, mocker):
        """explanation dict must contain 'personalization_boost' and 'original_score'."""
        from src.recommender.config import settings

        prefs = _make_prefs(
            language_preferences={"Python": 0.5},
            total_interactions=settings.min_interactions_for_personalization,
        )
        mocker.patch.object(
            db_manager, "get_user_preferences", new_callable=AsyncMock, return_value=prefs
        )
        engine = _make_engine()
        result = _make_result("a/a", language="Python", score=0.6, rank=1)

        results = await engine._apply_personalization([result], _make_request())

        expl = results[0].explanation
        assert "personalization_boost" in expl
        assert "original_score" in expl
        assert expl["original_score"] == pytest.approx(0.6)

    async def test_explanation_personalized_flag_set_to_true(self, mocker):
        from src.recommender.config import settings

        prefs = _make_prefs(
            language_preferences={"Python": 0.5},
            total_interactions=settings.min_interactions_for_personalization,
        )
        mocker.patch.object(
            db_manager, "get_user_preferences", new_callable=AsyncMock, return_value=prefs
        )
        engine = _make_engine()
        result = _make_result("a/a", language="Python", score=0.6, rank=1)

        results = await engine._apply_personalization([result], _make_request())

        assert results[0].explanation["personalized"] is True


# ---------------------------------------------------------------------------
# TestRecommend
# ---------------------------------------------------------------------------


class TestRecommend:
    def _patch_hybrid_recommend(self, mocker, return_value=None):
        """Patch the parent hybrid recommend so we only test the personalization layer."""
        if return_value is None:
            return_value = []
        return mocker.patch(
            "src.recommender.engines.personalized.HybridRetrievalEngine.recommend",
            new_callable=AsyncMock,
            return_value=return_value,
        )

    async def test_recommend_skips_personalization_when_disabled_in_settings(self, mocker):
        base_results = [_make_result("a/a", score=0.9, rank=1)]
        self._patch_hybrid_recommend(mocker, return_value=base_results)
        mocker.patch(
            "src.recommender.engines.personalized.settings",
            enable_personalization=False,
            min_interactions_for_personalization=5,
            personalization_weight=0.15,
            hybrid_search_top_k=100,
            rerank_top_k=20,
        )
        mock_apply = mocker.patch.object(
            PersonalizedEngine, "_apply_personalization", new_callable=AsyncMock
        )
        engine = _make_engine()
        await engine.recommend(_make_request())

        mock_apply.assert_not_awaited()

    async def test_recommend_skips_personalization_when_no_user_id(self, mocker):
        base_results = [_make_result("a/a", score=0.9, rank=1)]
        self._patch_hybrid_recommend(mocker, return_value=base_results)
        mock_apply = mocker.patch.object(
            PersonalizedEngine, "_apply_personalization", new_callable=AsyncMock
        )
        engine = _make_engine()
        # No user_id in request
        await engine.recommend(_make_request(user_id=None))

        mock_apply.assert_not_awaited()

    async def test_recommend_skips_personalization_when_flag_false_in_request(self, mocker):
        base_results = [_make_result("a/a", score=0.9, rank=1)]
        self._patch_hybrid_recommend(mocker, return_value=base_results)
        mock_apply = mocker.patch.object(
            PersonalizedEngine, "_apply_personalization", new_callable=AsyncMock
        )
        engine = _make_engine()
        await engine.recommend(_make_request(enable_personalization=False))

        mock_apply.assert_not_awaited()

    async def test_recommend_applies_personalization_with_valid_user_and_prefs(self, mocker):
        from src.recommender.config import settings as real_settings

        base_results = [_make_result("a/a", language="Python", score=0.5, rank=1)]
        self._patch_hybrid_recommend(mocker, return_value=base_results)
        prefs = _make_prefs(
            language_preferences={"Python": 0.8},
            total_interactions=real_settings.min_interactions_for_personalization,
        )
        mocker.patch.object(
            db_manager, "get_user_preferences", new_callable=AsyncMock, return_value=prefs
        )
        engine = _make_engine()
        results = await engine.recommend(
            _make_request(user_id="user-42", enable_personalization=True)
        )

        # Score should have been boosted
        assert results[0].score > 0.5


# ---------------------------------------------------------------------------
# TestExplain
# ---------------------------------------------------------------------------


class TestExplain:
    async def test_explain_includes_parent_explanation(self, mocker):
        mocker.patch.object(
            db_manager, "get_user_preferences", new_callable=AsyncMock, return_value=None
        )
        engine = _make_engine()
        result = await engine.explain("owner/repo", _make_request())

        # Inherits hybrid keys
        assert "method" in result
        assert "query" in result

    async def test_explain_adds_personalization_key(self, mocker):
        mocker.patch.object(
            db_manager, "get_user_preferences", new_callable=AsyncMock, return_value=None
        )
        engine = _make_engine()
        result = await engine.explain("owner/repo", _make_request())

        assert "personalization" in result

    async def test_explain_includes_user_interactions_count_when_prefs_found(self, mocker):
        prefs = _make_prefs(total_interactions=42)
        mocker.patch.object(
            db_manager, "get_user_preferences", new_callable=AsyncMock, return_value=prefs
        )
        engine = _make_engine()
        result = await engine.explain("owner/repo", _make_request(user_id="user-42"))

        assert result.get("user_interactions") == 42

    async def test_explain_includes_top_languages(self, mocker):
        prefs = _make_prefs(
            language_preferences={"Python": 0.9, "Rust": 0.6, "Go": 0.4},
            total_interactions=20,
        )
        mocker.patch.object(
            db_manager, "get_user_preferences", new_callable=AsyncMock, return_value=prefs
        )
        engine = _make_engine()
        result = await engine.explain("owner/repo", _make_request(user_id="user-42"))

        top_langs = result.get("top_languages")
        assert top_langs is not None
        assert isinstance(top_langs, dict)
        assert "Python" in top_langs
