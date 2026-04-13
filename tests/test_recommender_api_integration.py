"""
API integration tests for the recommender FastAPI service.
These tests require live Qdrant, MongoDB, and Redis connections.
Run with:
    pytest tests/test_recommender_api_integration.py -m integration -v -s
"""

import asyncio
import os
import pytest
from datetime import datetime, timezone

# Load environment variables BEFORE any src.* imports so that database
# connection settings, model paths, and service URLs are available at
# import time (some modules read them at module level).
from dotenv import load_dotenv

load_dotenv()

from httpx import AsyncClient, ASGITransport  # noqa: E402

from src.recommender.api import app  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def event_loop():
    """Module-scoped event loop — required so the module-scoped async client
    fixture doesn't clash with pytest-asyncio's default function-scoped loop."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
async def client():
    """
    Create a single AsyncClient that shares one lifespan across the whole
    test module.  Using scope="module" means the FastAPI startup (DB
    connections, model loading, engine initialisation) runs exactly once,
    keeping the test suite fast and side-effect-free between individual tests.

    If startup raises an exception the fixture re-raises it as a pytest.skip
    so every test in the module is skipped with an informative message rather
    than crashing with an opaque error.
    """
    transport = ASGITransport(app=app)
    try:
        async with AsyncClient(transport=transport, base_url="http://test") as async_client:
            # ASGITransport can swallow lifespan errors.  If startup didn't
            # finish (databases unreachable), required state won't be set.
            if not hasattr(app.state, "ab_test_service"):
                pytest.skip(
                    "Recommender startup incomplete — databases (MongoDB/Redis) "
                    "are not reachable from the host machine."
                )
            yield async_client
    except Exception as exc:
        pytest.skip(f"Recommender service startup failed — live databases unavailable.\nDetails: {exc}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_top_results(query: str, results: list, n: int = 3) -> None:
    """Print top N results so a developer can eyeball recommendation quality."""
    print(f"\n[{query}] Top results:")
    for r in results[:n]:
        print(f"  {r['rank']}. {r['name']} (score={r['score']:.3f}, stars={r.get('stars', '?')})")


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.api
class TestRecommenderAPI:
    """End-to-end API integration tests against live databases."""

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_health_returns_200(self, client: AsyncClient) -> None:
        """GET /health must return HTTP 200 with the expected body shape."""
        response = await client.get("/health")

        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "recommender"
        assert "version" in data
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_health_timestamp_is_iso_format(self, client: AsyncClient) -> None:
        """The timestamp field in the health response must be a valid ISO 8601 string."""
        response = await client.get("/health")
        assert response.status_code == 200

        timestamp_str = response.json()["timestamp"]
        # datetime.fromisoformat raises ValueError when the string is invalid.
        parsed = datetime.fromisoformat(timestamp_str)
        assert isinstance(parsed, datetime)

    # ------------------------------------------------------------------
    # /recommend — basic shape
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_recommend_returns_200_for_valid_query(self, client: AsyncClient) -> None:
        """POST /recommend with a minimal valid payload must return HTTP 200."""
        payload = {"query": "machine learning python", "top_k": 5}
        response = await client.post("/recommend", json=payload)

        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_recommend_response_has_correct_shape(self, client: AsyncClient) -> None:
        """The response body must contain all required top-level fields with correct types."""
        payload = {"query": "machine learning python", "top_k": 5}
        response = await client.post("/recommend", json=payload)
        assert response.status_code == 200

        data = response.json()

        assert "query" in data
        assert isinstance(data["results"], list)
        assert isinstance(data["total_candidates"], int) and data["total_candidates"] >= 0
        assert isinstance(data["processing_time_ms"], float) and data["processing_time_ms"] > 0
        assert isinstance(data["variant"], str)
        assert isinstance(data["personalized"], bool)
        assert isinstance(data["filters_applied"], dict)

    @pytest.mark.asyncio
    async def test_recommend_results_have_correct_fields(self, client: AsyncClient) -> None:
        """Every entry in `results` must expose repo_id, name, score, rank, and explanation."""
        payload = {"query": "machine learning python", "top_k": 5}
        response = await client.post("/recommend", json=payload)
        assert response.status_code == 200

        results = response.json()["results"]
        _print_top_results("machine learning python", results)

        for result in results:
            assert "repo_id" in result, f"Missing repo_id in {result}"
            assert "name" in result, f"Missing name in {result}"
            assert "score" in result and result["score"] > 0, f"score must be > 0, got {result.get('score')}"
            assert "rank" in result and result["rank"] > 0, f"rank must be > 0, got {result.get('rank')}"
            assert "explanation" in result and isinstance(result["explanation"], dict), (
                f"explanation must be a dict, got {result.get('explanation')}"
            )

    # ------------------------------------------------------------------
    # /recommend — top_k, diversity, filters
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_recommend_top_k_respected(self, client: AsyncClient) -> None:
        """The number of returned results must not exceed the requested top_k."""
        payload = {"query": "machine learning python", "top_k": 3}
        response = await client.post("/recommend", json=payload)
        assert response.status_code == 200

        results = response.json()["results"]
        assert len(results) <= 3, f"Expected at most 3 results for top_k=3, got {len(results)}"

    @pytest.mark.asyncio
    async def test_recommend_different_queries_give_different_results(self, client: AsyncClient) -> None:
        """Two semantically unrelated queries must not return the same top result."""
        payload_ml = {"query": "machine learning", "top_k": 5}
        payload_k8s = {"query": "kubernetes docker", "top_k": 5}

        response_ml = await client.post("/recommend", json=payload_ml)
        response_k8s = await client.post("/recommend", json=payload_k8s)

        assert response_ml.status_code == 200
        assert response_k8s.status_code == 200

        results_ml = response_ml.json()["results"]
        results_k8s = response_k8s.json()["results"]

        _print_top_results("machine learning", results_ml)
        _print_top_results("kubernetes docker", results_k8s)

        # If either query returns no results we cannot compare — skip gracefully.
        if not results_ml or not results_k8s:
            pytest.skip("One or both queries returned no results; skipping diversity check.")

        top_ml = results_ml[0]["repo_id"]
        top_k8s = results_k8s[0]["repo_id"]
        assert top_ml != top_k8s, f"Expected different top results for different queries but both returned '{top_ml}'"

    @pytest.mark.asyncio
    async def test_recommend_with_language_filter(self, client: AsyncClient) -> None:
        """Results with a non-null language field must match the requested language filter."""
        payload = {"query": "web framework", "language": "Python", "top_k": 10}
        response = await client.post("/recommend", json=payload)
        assert response.status_code == 200

        results = response.json()["results"]
        _print_top_results("web framework [language=Python]", results)

        for result in results:
            lang = result.get("language")
            if lang is not None:
                assert lang == "Python", f"Expected language 'Python' but got '{lang}' for repo {result['name']}"

    @pytest.mark.asyncio
    async def test_recommend_with_min_stars_filter(self, client: AsyncClient) -> None:
        """Every returned repository must have at least min_stars stars."""
        min_stars = 100
        payload = {"query": "machine learning", "min_stars": min_stars, "top_k": 10}
        response = await client.post("/recommend", json=payload)
        assert response.status_code == 200

        results = response.json()["results"]
        _print_top_results(f"machine learning [min_stars={min_stars}]", results)

        for result in results:
            assert result.get("stars", 0) >= min_stars, (
                f"Repo '{result['name']}' has {result.get('stars')} stars, expected >= {min_stars}"
            )

    # ------------------------------------------------------------------
    # /recommend — variant
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_recommend_with_variant_baseline(self, client: AsyncClient) -> None:
        """Requesting variant='baseline' must result in variant='baseline' in the response."""
        payload = {"query": "python", "variant": "baseline", "top_k": 5}
        response = await client.post("/recommend", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["variant"] == "baseline", f"Expected variant 'baseline', got '{data['variant']}'"

    # ------------------------------------------------------------------
    # /recommend — performance
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_recommend_processing_time_is_reasonable(self, client: AsyncClient) -> None:
        """The reported processing_time_ms must be under 30 000 ms (30 seconds)."""
        payload = {"query": "machine learning python", "top_k": 5}
        response = await client.post("/recommend", json=payload)
        assert response.status_code == 200

        processing_time_ms = response.json()["processing_time_ms"]
        assert processing_time_ms < 30_000, f"Processing time {processing_time_ms:.1f} ms exceeded 30 000 ms threshold"

    # ------------------------------------------------------------------
    # /recommend/explain/{repo_id}
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_explain_returns_explanation(self, client: AsyncClient) -> None:
        """POST /recommend/explain/{repo_id} must return a dict with 'engine' and 'method'."""
        repo_id = "some-repo-id"
        payload = {"query": "machine learning"}
        response = await client.post(f"/recommend/explain/{repo_id}", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, dict), f"Expected dict response, got {type(data)}"
        assert "engine" in data, f"Missing 'engine' key in explanation: {data}"
        assert "method" in data, f"Missing 'method' key in explanation: {data}"

    # ------------------------------------------------------------------
    # /interaction
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_log_interaction_returns_success(self, client: AsyncClient) -> None:
        """POST /interaction with a valid body must return status='success' and interaction_id."""
        payload = {
            "user_id": "test-user-integration",
            "query": "machine learning python",
            "repo_id": "test-repo-123",
            "interaction_type": "view",
            "variant": "baseline",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        response = await client.post("/interaction", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data.get("status") == "success", f"Expected status='success', got: {data}"
        assert "interaction_id" in data, f"Missing 'interaction_id' in response: {data}"

    # ------------------------------------------------------------------
    # /preferences/{user_id}
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_preferences_returns_404_for_unknown_user(self, client: AsyncClient) -> None:
        """GET /preferences/{user_id} must return 404 when the user is not found."""
        response = await client.get("/preferences/nonexistent-user-xyz")
        assert response.status_code == 404

    # ------------------------------------------------------------------
    # /ab-test
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_ab_test_endpoint_returns_valid_response(self, client: AsyncClient) -> None:
        """GET /ab-test must return HTTP 200 and a non-empty dict."""
        response = await client.get("/ab-test")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, dict), f"Expected dict, got {type(data)}"
        # Either it has a 'status' key (no active test) or A/B test fields.
        assert len(data) > 0, "Response dict must not be empty"

    # ------------------------------------------------------------------
    # /admin/engines
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_admin_engines_lists_all_engines(self, client: AsyncClient) -> None:
        """GET /admin/engines must list baseline, hybrid, and personalized engines."""
        response = await client.get("/admin/engines")
        assert response.status_code == 200

        data = response.json()
        assert "engines" in data, f"Missing 'engines' key in response: {data}"

        engines = data["engines"]
        assert isinstance(engines, list), f"Expected list for 'engines', got {type(engines)}"

        engine_names = {e.get("name") for e in engines}
        for expected in ("baseline", "hybrid", "personalized"):
            assert expected in engine_names, f"Engine '{expected}' not found in engines list. Got: {engine_names}"

    # ------------------------------------------------------------------
    # /admin/models
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_admin_models_returns_list(self, client: AsyncClient) -> None:
        """GET /admin/models must return HTTP 200 with a 'models' key (list, may be empty)."""
        response = await client.get("/admin/models")
        assert response.status_code == 200

        data = response.json()
        assert "models" in data, f"Missing 'models' key in response: {data}"
        assert isinstance(data["models"], list), f"Expected list for 'models', got {type(data['models'])}"

    # ------------------------------------------------------------------
    # /admin/cache/clear
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_admin_cache_clear_returns_success(self, client: AsyncClient) -> None:
        """POST /admin/cache/clear must return status='success'."""
        response = await client.post("/admin/cache/clear")
        assert response.status_code == 200

        data = response.json()
        assert data.get("status") == "success", f"Expected status='success', got: {data}"

    # ------------------------------------------------------------------
    # Validation errors
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_recommend_invalid_query_handled(self, client: AsyncClient) -> None:
        """POST /recommend without 'query' must return HTTP 422 (validation error)."""
        response = await client.post("/recommend", json={"top_k": 5})
        assert response.status_code == 422

    # ------------------------------------------------------------------
    # /metrics/{variant}
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_metrics_returns_404_for_unknown_variant(self, client: AsyncClient) -> None:
        """GET /metrics/{variant} must return 404 when no metrics exist for that variant."""
        response = await client.get("/metrics/nonexistent-variant")
        assert response.status_code == 404
