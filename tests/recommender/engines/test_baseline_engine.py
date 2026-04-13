"""Unit tests for BaselineEngine (London School TDD — mock-first)."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from src.recommender.database import db_manager
from src.recommender.engines.baseline import BaselineEngine
from src.recommender.models import RecommendationRequest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(
    query: str = "machine learning",
    top_k: int = 5,
    language: str | None = None,
    min_stars: int | None = None,
    license: str | None = None,
) -> RecommendationRequest:
    return RecommendationRequest(
        query=query,
        top_k=top_k,
        language=language,
        min_stars=min_stars,
        license=license,
    )


def _make_repo(
    repo_id: str = "owner/repo",
    name: str = "repo",
    full_name: str = "owner/repo",
    language: str = "Python",
    stars: int = 100,
    license: str = "MIT",
) -> dict:
    return {
        "repo_id": repo_id,
        "name": name,
        "full_name": full_name,
        "description": "A test repo",
        "language": language,
        "stars": stars,
        "forks": 10,
        "url": f"https://github.com/{full_name}",
        "license": license,
        "last_updated": datetime.now(timezone.utc),
    }


# ---------------------------------------------------------------------------
# TestRecommend
# ---------------------------------------------------------------------------


class TestRecommend:
    async def test_recommend_returns_repository_results(self, mocker):
        mock_search = mocker.patch.object(
            db_manager,
            "search_repositories",
            new_callable=AsyncMock,
            return_value=[_make_repo("owner/alpha")],
        )
        engine = BaselineEngine()
        results = await engine.recommend(_make_request(top_k=5))

        assert len(results) == 1
        assert results[0].repo_id == "owner/alpha"
        mock_search.assert_awaited_once()

    async def test_recommend_uses_text_operator_not_regex(self, mocker):
        mock_search = mocker.patch.object(
            db_manager,
            "search_repositories",
            new_callable=AsyncMock,
            return_value=[],
        )
        engine = BaselineEngine()
        await engine.recommend(_make_request(query="fast api"))

        call_kwargs = mock_search.call_args.kwargs
        query_filter = call_kwargs.get(
            "query_filter", mock_search.call_args.args[0] if mock_search.call_args.args else {}
        )
        assert "$text" in query_filter, "must use $text operator"
        assert query_filter["$text"] == {"$search": "fast api"}
        assert "$regex" not in str(query_filter), "must NOT use $regex"

    async def test_recommend_respects_top_k(self, mocker):
        repos = [_make_repo(f"o/r{i}") for i in range(10)]
        mock_search = mocker.patch.object(
            db_manager,
            "search_repositories",
            new_callable=AsyncMock,
            return_value=repos,
        )
        engine = BaselineEngine()
        await engine.recommend(_make_request(top_k=3))

        call_kwargs = mock_search.call_args
        limit_passed = call_kwargs.kwargs.get("limit") or (call_kwargs.args[1] if len(call_kwargs.args) > 1 else None)
        assert limit_passed == 3

    async def test_recommend_applies_language_filter(self, mocker):
        mock_search = mocker.patch.object(
            db_manager,
            "search_repositories",
            new_callable=AsyncMock,
            return_value=[],
        )
        engine = BaselineEngine()
        await engine.recommend(_make_request(language="Rust"))

        call_kwargs = mock_search.call_args.kwargs
        query_filter = call_kwargs.get(
            "query_filter", mock_search.call_args.args[0] if mock_search.call_args.args else {}
        )
        assert query_filter.get("language") == "Rust"

    async def test_recommend_applies_min_stars_filter(self, mocker):
        mock_search = mocker.patch.object(
            db_manager,
            "search_repositories",
            new_callable=AsyncMock,
            return_value=[],
        )
        engine = BaselineEngine()
        await engine.recommend(_make_request(min_stars=500))

        call_kwargs = mock_search.call_args.kwargs
        query_filter = call_kwargs.get(
            "query_filter", mock_search.call_args.args[0] if mock_search.call_args.args else {}
        )
        assert query_filter.get("stars") == {"$gte": 500}

    async def test_recommend_applies_license_filter(self, mocker):
        mock_search = mocker.patch.object(
            db_manager,
            "search_repositories",
            new_callable=AsyncMock,
            return_value=[],
        )
        engine = BaselineEngine()
        await engine.recommend(_make_request(license="Apache-2.0"))

        call_kwargs = mock_search.call_args.kwargs
        query_filter = call_kwargs.get(
            "query_filter", mock_search.call_args.args[0] if mock_search.call_args.args else {}
        )
        assert query_filter.get("license") == "Apache-2.0"

    async def test_recommend_assigns_1_indexed_ranks(self, mocker):
        repos = [_make_repo(f"o/r{i}") for i in range(3)]
        mocker.patch.object(
            db_manager,
            "search_repositories",
            new_callable=AsyncMock,
            return_value=repos,
        )
        engine = BaselineEngine()
        results = await engine.recommend(_make_request(top_k=3))

        ranks = [r.rank for r in results]
        assert ranks == [1, 2, 3]

    async def test_recommend_score_decreases_with_rank(self, mocker):
        repos = [_make_repo(f"o/r{i}") for i in range(3)]
        mocker.patch.object(
            db_manager,
            "search_repositories",
            new_callable=AsyncMock,
            return_value=repos,
        )
        engine = BaselineEngine()
        results = await engine.recommend(_make_request(top_k=3))

        assert results[0].score == pytest.approx(1 / 1)
        assert results[1].score == pytest.approx(1 / 2)
        assert results[2].score == pytest.approx(1 / 3)

    async def test_recommend_returns_empty_when_no_repos_found(self, mocker):
        mocker.patch.object(
            db_manager,
            "search_repositories",
            new_callable=AsyncMock,
            return_value=[],
        )
        engine = BaselineEngine()
        results = await engine.recommend(_make_request())

        assert results == []

    async def test_recommend_with_empty_query(self, mocker):
        """No $text key in the filter when query is an empty string."""
        mock_search = mocker.patch.object(
            db_manager,
            "search_repositories",
            new_callable=AsyncMock,
            return_value=[],
        )
        engine = BaselineEngine()
        await engine.recommend(_make_request(query=""))

        call_kwargs = mock_search.call_args.kwargs
        query_filter = call_kwargs.get(
            "query_filter", mock_search.call_args.args[0] if mock_search.call_args.args else {}
        )
        assert "$text" not in query_filter


# ---------------------------------------------------------------------------
# TestExplain
# ---------------------------------------------------------------------------


class TestExplain:
    async def test_explain_returns_engine_name(self):
        engine = BaselineEngine()
        result = await engine.explain("owner/repo", _make_request())

        assert result["engine"] == "baseline"

    async def test_explain_returns_keyword_method(self):
        engine = BaselineEngine()
        result = await engine.explain("owner/repo", _make_request())

        assert result["method"] == "keyword_search"

    async def test_explain_includes_query(self):
        engine = BaselineEngine()
        request = _make_request(query="pytorch neural network")
        result = await engine.explain("owner/repo", request)

        assert result["query"] == "pytorch neural network"
