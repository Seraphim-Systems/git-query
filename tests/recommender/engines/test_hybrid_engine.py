"""Unit tests for HybridRetrievalEngine (London School TDD — mock-first)."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.recommender.database import db_manager
from src.recommender.engines.hybrid import HybridRetrievalEngine
from src.recommender.models import RecommendationRequest, RepositoryResult


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


def _make_repo_result(
    repo_id: str = "owner/repo",
    language: str | None = "Python",
    stars: int = 100,
    score: float = 0.5,
    rank: int = 1,
    license: str | None = "MIT",
) -> RepositoryResult:
    return RepositoryResult(
        repo_id=repo_id,
        name=repo_id.split("/")[-1],
        full_name=repo_id,
        description="A test repo",
        language=language,
        stars=stars,
        forks=5,
        url=f"https://github.com/{repo_id}",
        license=license,
        last_updated=datetime.now(timezone.utc),
        score=score,
        rank=rank,
        explanation={"method": "hybrid_rrf", "sources": ["semantic"], "rrf_score": score},
    )


def _make_engine(embedding_service=None, reranker_service=None) -> HybridRetrievalEngine:
    return HybridRetrievalEngine(
        embedding_service=embedding_service,
        reranker_service=reranker_service,
    )


def _mock_embedding_service() -> AsyncMock:
    svc = AsyncMock()
    svc.embed_text = AsyncMock(return_value=[0.1] * 384)
    return svc


# ---------------------------------------------------------------------------
# TestReciprocalRankFusion
# ---------------------------------------------------------------------------


class TestReciprocalRankFusion:
    def test_rrf_semantic_only_results(self):
        engine = _make_engine()
        semantic = [{"repo_id": "a/b", "score": 0.9, "source": "semantic", "payload": {}}]
        results = engine._reciprocal_rank_fusion(semantic, [])

        assert len(results) == 1
        assert results[0].repo_id == "a/b"
        assert results[0].score == pytest.approx(1 / (60 + 1))

    def test_rrf_keyword_only_results(self):
        engine = _make_engine()
        keyword = [
            {
                "repo_id": "c/d",
                "score": 1.0,
                "source": "keyword",
                "repo_data": {"name": "d", "full_name": "c/d", "stars": 50, "forks": 2, "url": "https://github.com/c/d"},
            }
        ]
        results = engine._reciprocal_rank_fusion([], keyword)

        assert len(results) == 1
        assert results[0].repo_id == "c/d"
        assert results[0].score == pytest.approx(1 / 61)

    def test_rrf_combines_both_sources(self):
        """A repo_id appearing in both sources accumulates a higher fused score."""
        engine = _make_engine()
        repo_id = "shared/repo"
        semantic = [{"repo_id": repo_id, "score": 0.8, "source": "semantic", "payload": {}}]
        keyword = [
            {
                "repo_id": repo_id,
                "score": 1.0,
                "source": "keyword",
                "repo_data": {"name": "repo", "full_name": repo_id, "stars": 10, "forks": 1, "url": ""},
            }
        ]

        results_combined = engine._reciprocal_rank_fusion(semantic, keyword)
        results_semantic_only = engine._reciprocal_rank_fusion(semantic, [])
        results_keyword_only = engine._reciprocal_rank_fusion([], keyword)

        combined_score = results_combined[0].score
        assert combined_score > results_semantic_only[0].score
        assert combined_score > results_keyword_only[0].score
        assert combined_score == pytest.approx(2 / 61)

    def test_rrf_score_formula(self):
        """Rank-1 result with k=60 yields score == 1/61."""
        engine = _make_engine()
        assert engine.k == 60
        semantic = [{"repo_id": "x/y", "score": 0.99, "source": "semantic", "payload": {}}]

        results = engine._reciprocal_rank_fusion(semantic, [])

        assert results[0].score == pytest.approx(1 / 61)

    def test_rrf_keyword_data_takes_precedence_over_payload(self):
        """When both sources carry metadata for the same repo, keyword data wins."""
        engine = _make_engine()
        repo_id = "owner/dual"
        semantic = [
            {
                "repo_id": repo_id,
                "score": 0.9,
                "source": "semantic",
                "payload": {"name": "from_payload", "stars": 1, "forks": 0, "url": "", "full_name": repo_id},
            }
        ]
        keyword = [
            {
                "repo_id": repo_id,
                "score": 1.0,
                "source": "keyword",
                "repo_data": {"name": "from_keyword", "full_name": repo_id, "stars": 999, "forks": 5, "url": "https://github.com/owner/dual"},
            }
        ]

        results = engine._reciprocal_rank_fusion(semantic, keyword)

        assert results[0].name == "from_keyword"
        assert results[0].stars == 999

    def test_rrf_falls_back_to_payload_when_no_keyword_data(self):
        """When only semantic data is present, payload fields are used."""
        engine = _make_engine()
        semantic = [
            {
                "repo_id": "p/q",
                "score": 0.7,
                "source": "semantic",
                "payload": {"name": "payload_name", "stars": 42, "forks": 3, "url": "", "full_name": "p/q"},
            }
        ]

        results = engine._reciprocal_rank_fusion(semantic, [])

        assert results[0].name == "payload_name"
        assert results[0].stars == 42

    def test_rrf_derives_name_from_repo_id_when_no_metadata(self):
        """With empty payload and no keyword data, name is derived from repo_id."""
        engine = _make_engine()
        semantic = [
            {"repo_id": "myorg/my-cool-repo", "score": 0.5, "source": "semantic", "payload": {}}
        ]

        results = engine._reciprocal_rank_fusion(semantic, [])

        assert results[0].name == "my-cool-repo"

    def test_rrf_sorts_by_fused_score_descending(self):
        """Results are returned highest fused score first."""
        engine = _make_engine()
        # repo_a appears only in semantic at rank 1, repo_b in both at rank 1 each
        semantic = [
            {"repo_id": "a/a", "score": 0.9, "source": "semantic", "payload": {}},
            {"repo_id": "b/b", "score": 0.8, "source": "semantic", "payload": {}},
        ]
        keyword = [
            {
                "repo_id": "b/b",
                "score": 1.0,
                "source": "keyword",
                "repo_data": {"name": "b", "full_name": "b/b", "stars": 5, "forks": 0, "url": ""},
            }
        ]

        results = engine._reciprocal_rank_fusion(semantic, keyword)

        # b/b should rank first because it appears in both lists
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert results[0].repo_id == "b/b"


# ---------------------------------------------------------------------------
# TestSemanticSearch
# ---------------------------------------------------------------------------


class TestSemanticSearch:
    async def test_semantic_search_awaits_vector_search_directly(self, mocker):
        """db_manager.vector_search must be called as an async call (no run_in_executor)."""
        mock_vector_search = mocker.patch.object(
            db_manager,
            "vector_search",
            new_callable=AsyncMock,
            return_value=[],
        )
        embedding_svc = _mock_embedding_service()
        engine = _make_engine(embedding_service=embedding_svc)

        await engine._semantic_search(_make_request())

        mock_vector_search.assert_awaited_once()

    async def test_semantic_search_returns_empty_when_no_embedding_service(self, mocker):
        mocker.patch.object(db_manager, "vector_search", new_callable=AsyncMock, return_value=[])
        engine = _make_engine(embedding_service=None)

        results = await engine._semantic_search(_make_request())

        assert results == []

    async def test_semantic_search_maps_payload_repo_id_over_qdrant_point_id(self, mocker):
        """The string repo_id from the payload is preferred over the Qdrant point UUID."""
        qdrant_point_id = "00000000-0000-0000-0000-000000000001"
        payload_repo_id = "owner/real-repo"
        mocker.patch.object(
            db_manager,
            "vector_search",
            new_callable=AsyncMock,
            return_value=[
                {
                    "repo_id": qdrant_point_id,
                    "score": 0.95,
                    "payload": {"repo_id": payload_repo_id, "stars": 50},
                }
            ],
        )
        mocker.patch.object(
            db_manager,
            "get_repositories_by_repo_ids",
            new_callable=AsyncMock,
            return_value={},
        )
        embedding_svc = _mock_embedding_service()
        engine = _make_engine(embedding_service=embedding_svc)

        results = await engine._semantic_search(_make_request())

        assert len(results) == 1
        assert results[0]["repo_id"] == payload_repo_id

    async def test_semantic_search_triggers_enrichment_for_repos_without_stars(self, mocker):
        """Repos with '/' in ID and no 'stars' in payload trigger get_repositories_by_repo_ids."""
        repo_id = "org/sparse-repo"
        mocker.patch.object(
            db_manager,
            "vector_search",
            new_callable=AsyncMock,
            return_value=[
                {"repo_id": repo_id, "score": 0.8, "payload": {"repo_id": repo_id}}
            ],
        )
        mock_enrich = mocker.patch.object(
            db_manager,
            "get_repositories_by_repo_ids",
            new_callable=AsyncMock,
            return_value={repo_id: {"stars": 123, "name": "sparse-repo", "full_name": repo_id}},
        )
        embedding_svc = _mock_embedding_service()
        engine = _make_engine(embedding_service=embedding_svc)

        results = await engine._semantic_search(_make_request())

        mock_enrich.assert_awaited_once_with([repo_id])
        assert results[0]["payload"]["stars"] == 123

    async def test_semantic_search_skips_enrichment_when_stars_present(self, mocker):
        """Repos whose payload already contains 'stars' do not trigger enrichment."""
        repo_id = "org/rich-repo"
        mocker.patch.object(
            db_manager,
            "vector_search",
            new_callable=AsyncMock,
            return_value=[
                {"repo_id": repo_id, "score": 0.8, "payload": {"repo_id": repo_id, "stars": 500}}
            ],
        )
        mock_enrich = mocker.patch.object(
            db_manager,
            "get_repositories_by_repo_ids",
            new_callable=AsyncMock,
            return_value={},
        )
        embedding_svc = _mock_embedding_service()
        engine = _make_engine(embedding_service=embedding_svc)

        await engine._semantic_search(_make_request())

        mock_enrich.assert_not_awaited()


# ---------------------------------------------------------------------------
# TestKeywordSearch
# ---------------------------------------------------------------------------


class TestKeywordSearch:
    async def test_keyword_search_uses_text_operator(self, mocker):
        mock_search = mocker.patch.object(
            db_manager,
            "search_repositories",
            new_callable=AsyncMock,
            return_value=[],
        )
        engine = _make_engine()
        await engine._keyword_search(_make_request(query="kubernetes operator"))

        call_kwargs = mock_search.call_args.kwargs
        query_filter = call_kwargs.get(
            "query_filter", mock_search.call_args.args[0] if mock_search.call_args.args else {}
        )
        assert "$text" in query_filter
        assert query_filter["$text"] == {"$search": "kubernetes operator"}

    async def test_keyword_search_returns_empty_when_no_repos(self, mocker):
        mocker.patch.object(
            db_manager,
            "search_repositories",
            new_callable=AsyncMock,
            return_value=[],
        )
        engine = _make_engine()
        results = await engine._keyword_search(_make_request())

        assert results == []


# ---------------------------------------------------------------------------
# TestApplyFilters
# ---------------------------------------------------------------------------


class TestApplyFilters:
    def test_apply_filters_language_excludes_wrong_language(self):
        engine = _make_engine()
        results = [
            _make_repo_result("a/a", language="Python"),
            _make_repo_result("b/b", language="Go"),
        ]
        request = _make_request(language="Python")
        filtered = engine._apply_filters(results, request)

        assert len(filtered) == 1
        assert filtered[0].repo_id == "a/a"

    def test_apply_filters_min_stars_excludes_below_threshold(self):
        engine = _make_engine()
        results = [
            _make_repo_result("a/a", stars=1000),
            _make_repo_result("b/b", stars=10),
        ]
        request = _make_request(min_stars=500)
        filtered = engine._apply_filters(results, request)

        assert len(filtered) == 1
        assert filtered[0].repo_id == "a/a"

    def test_apply_filters_license_excludes_wrong_license(self):
        engine = _make_engine()
        results = [
            _make_repo_result("a/a", license="MIT"),
            _make_repo_result("b/b", license="GPL-3.0"),
        ]
        request = _make_request(license="MIT")
        filtered = engine._apply_filters(results, request)

        assert len(filtered) == 1
        assert filtered[0].repo_id == "a/a"

    def test_apply_filters_no_filters_returns_all(self):
        engine = _make_engine()
        results = [
            _make_repo_result("a/a"),
            _make_repo_result("b/b"),
            _make_repo_result("c/c"),
        ]
        request = _make_request()
        filtered = engine._apply_filters(results, request)

        assert len(filtered) == 3

    def test_apply_filters_combined_filters(self):
        engine = _make_engine()
        results = [
            _make_repo_result("a/a", language="Python", stars=600, license="MIT"),
            _make_repo_result("b/b", language="Python", stars=200, license="MIT"),  # below min_stars
            _make_repo_result("c/c", language="Rust", stars=800, license="MIT"),    # wrong language
            _make_repo_result("d/d", language="Python", stars=700, license="GPL"),  # wrong license
        ]
        request = _make_request(language="Python", min_stars=500, license="MIT")
        filtered = engine._apply_filters(results, request)

        assert len(filtered) == 1
        assert filtered[0].repo_id == "a/a"


# ---------------------------------------------------------------------------
# TestRecommend (integration of all steps)
# ---------------------------------------------------------------------------


class TestRecommend:
    async def test_recommend_calls_reranker_when_results_exist(self, mocker):
        mocker.patch.object(
            db_manager, "vector_search", new_callable=AsyncMock, return_value=[]
        )
        mocker.patch.object(
            db_manager,
            "search_repositories",
            new_callable=AsyncMock,
            return_value=[
                {
                    "repo_id": f"o/r{i}",
                    "name": f"r{i}",
                    "full_name": f"o/r{i}",
                    "description": "",
                    "language": "Python",
                    "stars": 100,
                    "forks": 5,
                    "url": "",
                    "license": "MIT",
                    "last_updated": datetime.now(timezone.utc),
                }
                for i in range(5)
            ],
        )
        reranked_results = [_make_repo_result(f"o/r{i}", rank=i + 1, score=0.9 - i * 0.1) for i in range(3)]
        mock_reranker = AsyncMock()
        mock_reranker.rerank = AsyncMock(return_value=reranked_results)

        engine = _make_engine(reranker_service=mock_reranker)
        await engine.recommend(_make_request(top_k=5))

        mock_reranker.rerank.assert_awaited_once()

    async def test_recommend_skips_reranker_when_service_is_none(self, mocker):
        mocker.patch.object(
            db_manager, "vector_search", new_callable=AsyncMock, return_value=[]
        )
        mocker.patch.object(
            db_manager, "search_repositories", new_callable=AsyncMock, return_value=[]
        )
        engine = _make_engine(reranker_service=None)
        # Should not raise even without a reranker
        results = await engine.recommend(_make_request())
        assert isinstance(results, list)

    async def test_recommend_returns_top_k_results(self, mocker):
        mocker.patch.object(
            db_manager, "vector_search", new_callable=AsyncMock, return_value=[]
        )
        mocker.patch.object(
            db_manager,
            "search_repositories",
            new_callable=AsyncMock,
            return_value=[
                {
                    "repo_id": f"o/r{i}",
                    "name": f"r{i}",
                    "full_name": f"o/r{i}",
                    "description": "",
                    "language": "Python",
                    "stars": 100,
                    "forks": 5,
                    "url": "",
                    "license": "MIT",
                    "last_updated": datetime.now(timezone.utc),
                }
                for i in range(20)
            ],
        )
        engine = _make_engine()
        results = await engine.recommend(_make_request(top_k=3))

        assert len(results) == 3

    async def test_recommend_assigns_final_ranks(self, mocker):
        mocker.patch.object(
            db_manager, "vector_search", new_callable=AsyncMock, return_value=[]
        )
        mocker.patch.object(
            db_manager,
            "search_repositories",
            new_callable=AsyncMock,
            return_value=[
                {
                    "repo_id": f"o/r{i}",
                    "name": f"r{i}",
                    "full_name": f"o/r{i}",
                    "description": "",
                    "language": "Python",
                    "stars": 100,
                    "forks": 5,
                    "url": "",
                    "license": "MIT",
                    "last_updated": datetime.now(timezone.utc),
                }
                for i in range(4)
            ],
        )
        engine = _make_engine()
        results = await engine.recommend(_make_request(top_k=4))

        for idx, result in enumerate(results):
            assert result.rank == idx + 1


# ---------------------------------------------------------------------------
# TestExplain
# ---------------------------------------------------------------------------


class TestExplain:
    async def test_explain_returns_hybrid_method(self):
        engine = _make_engine()
        result = await engine.explain("owner/repo", _make_request())

        assert result["method"] == "hybrid_retrieval_rrf"
