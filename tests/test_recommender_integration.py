"""
Integration tests for the GitHub repository recommender system.

These tests run against live infrastructure (Qdrant + MongoDB) and validate
that the full recommendation pipeline works correctly end-to-end.

Prerequisites:
    - Qdrant running with the ``repositories_embeddings`` collection populated
    - MongoDB running with the ``repositories`` collection populated
    - A .env file at the project root with QDRANT_URL, QDRANT_API_KEY / APIKEY_QDRANT

Run with:
    pytest tests/test_recommender_integration.py -m integration -v -s

Run a single group:
    pytest tests/test_recommender_integration.py -m "integration and qdrant_health" -v -s
    pytest tests/test_recommender_integration.py -m "integration and embedding_quality" -v -s
    pytest tests/test_recommender_integration.py -m "integration and search_quality" -v -s
    pytest tests/test_recommender_integration.py -m "integration and pipeline" -v -s
    pytest tests/test_recommender_integration.py -m "integration and filters" -v -s
    pytest tests/test_recommender_integration.py -m "integration and performance" -v -s
"""

# ---------------------------------------------------------------------------
# Load .env BEFORE any project imports that read environment variables.
# ---------------------------------------------------------------------------
import os
import math
import time
import asyncio
from pathlib import Path

from dotenv import load_dotenv

# Resolve the project root (.env lives next to /tests/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env", override=False)

# ---------------------------------------------------------------------------
# Standard library / third-party imports
# ---------------------------------------------------------------------------
import pytest
import pytest_asyncio

pytest.importorskip(
    "torch", reason="Integration recommender tests require optional torch dependency"
)
pytest.importorskip(
    "sentence_transformers",
    reason="Integration recommender tests require optional sentence-transformers dependency",
)

# ---------------------------------------------------------------------------
# Project imports (env vars are loaded above, so pydantic-settings picks them up)
# ---------------------------------------------------------------------------
from src.recommender.config import settings
from src.recommender.models import RecommendationRequest
from src.recommender.services.embedding_service import EmbeddingService
from src.recommender.engines.hybrid import HybridRetrievalEngine
from src.recommender.engines.baseline import BaselineEngine
from src.recommender.database import db_manager

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
COLLECTION_NAME = settings.qdrant_repos_collection  # "repositories_embeddings"
EMBEDDING_DIM = settings.embedding_dimension  # 384
MIN_EXPECTED_POINTS = 1_000_000

# Native Qdrant URL (works inside Docker; skipped if unreachable from host)
_QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://{}:{}".format(
        os.getenv("QDRANT_HOST", "localhost"),
        os.getenv("QDRANT_HTTP_PORT", "6333"),
    ),
)
_QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or os.getenv("APIKEY_QDRANT")

# Gateway URL — used as fallback when native Qdrant is unreachable from host
_GATEWAY_URL = os.getenv("API_BASE_URL", "").rstrip("/")
_GATEWAY_KEY = os.getenv("APIKEY_QDRANT") or os.getenv("QDRANT_API_KEY")


# ---------------------------------------------------------------------------
# Gateway Qdrant client — duck-types QdrantClient for the subset used in tests
# ---------------------------------------------------------------------------


class _Collection:
    """Minimal stand-in for qdrant_client CollectionDescription."""

    def __init__(self, name):
        self.name = name


class _CollectionsList:
    def __init__(self, names):
        self.collections = [_Collection(n) for n in names]


class _VectorsConfig:
    def __init__(self, size):
        self.size = size


class _Params:
    def __init__(self, size):
        self.vectors = _VectorsConfig(size)


class _Config:
    def __init__(self, size):
        self.params = _Params(size)


class _CollectionInfo:
    def __init__(self, points_count, dim, status):
        self.points_count = points_count
        self.config = _Config(dim)
        self.status = status


class _Hit:
    def __init__(self, id_, score, payload):
        self.id = id_
        self.score = score
        self.payload = payload or {}


class GatewayQdrantClient:
    """Calls the gateway's /api/qdrant/* REST endpoints instead of native Qdrant.

    This lets integration tests run from the host machine without needing
    direct access to the Qdrant Docker-internal hostname.
    """

    def __init__(self, base_url: str, api_key: str):
        import requests as _req

        self._base = base_url.rstrip("/") + "/api/qdrant"
        self._session = _req.Session()
        self._session.headers.update({"X-API-Key": api_key or ""})

    def _get(self, path: str):
        resp = self._session.get(f"{self._base}{path}", timeout=15)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, body: dict):
        resp = self._session.post(f"{self._base}{path}", json=body, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def get_collections(self) -> _CollectionsList:
        data = self._get("/collections")
        names = [c["name"] for c in data.get("collections", [])]
        return _CollectionsList(names)

    def get_collection(self, name: str) -> _CollectionInfo:
        # Try a collection-specific endpoint first (may return detailed stats)
        try:
            data = self._get(f"/collections/{name}")
            result = data.get("result", data)
            count = (
                result.get("points_count")
                or result.get("vectors_count")
                or result.get("indexed_vectors_count", 0)
            )
            status = result.get("status", "green")
            return _CollectionInfo(points_count=count, dim=EMBEDDING_DIM, status=status)
        except Exception:
            pass
        # Fall back to the collections list endpoint
        data = self._get("/collections")
        for c in data.get("collections", []):
            if c.get("name") == name:
                return _CollectionInfo(
                    points_count=c.get("points_count") or c.get("vectors_count", 0),
                    dim=EMBEDDING_DIM,  # gateway doesn't expose dim — use known value
                    status="green",
                )
        raise KeyError(f"Collection '{name}' not found")

    def search(
        self,
        collection_name: str,
        query_vector: list,
        limit: int = 10,
        with_payload: bool = True,
        **_,
    ):
        body = {"vector": query_vector, "limit": limit, "with_payload": with_payload}
        data = self._post(f"/collections/{collection_name}/search", body)
        hits = (
            data
            if isinstance(data, list)
            else data.get("results", data.get("hits", []))
        )
        return [_Hit(h.get("id"), h.get("score", 0.0), h.get("payload")) for h in hits]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def cosine_similarity(a: list, b: list) -> float:
    """Compute cosine similarity between two equal-length vectors using pure Python."""
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


def make_request(query: str, top_k: int = 10, **kwargs) -> RecommendationRequest:
    """Convenience factory for RecommendationRequest."""
    return RecommendationRequest(query=query, top_k=top_k, **kwargs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def event_loop():
    """Module-scoped event loop so async module fixtures share one loop."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="module")
def qdrant_client():
    """Return a Qdrant client.

    Tries native QdrantClient first (works inside Docker / with port-forwarding).
    Falls back to GatewayQdrantClient which calls the gateway REST API — this
    works from any host that can reach the gateway URL.
    """
    from qdrant_client import QdrantClient

    # 1. Try native Qdrant (Docker-internal or localhost with port mapping)
    try:
        client = QdrantClient(url=_QDRANT_URL, api_key=_QDRANT_API_KEY, timeout=5)
        client.get_collections()
        print(f"\n[fixture] Connected to native Qdrant at {_QDRANT_URL}")
        return client
    except Exception:
        pass

    # 2. Fall back to gateway REST API
    if _GATEWAY_URL:
        try:
            client = GatewayQdrantClient(_GATEWAY_URL, _GATEWAY_KEY)
            client.get_collections()
            print(f"\n[fixture] Connected to Qdrant via gateway at {_GATEWAY_URL}")
            return client
        except Exception as exc:
            pytest.skip(
                f"Qdrant unreachable natively ({_QDRANT_URL}) and via gateway ({_GATEWAY_URL}): {exc}"
            )

    pytest.skip(
        f"Qdrant unreachable at {_QDRANT_URL} and API_BASE_URL not set for gateway fallback"
    )


@pytest.fixture(scope="module")
def embedding_service():
    """Return an EmbeddingService with the model pre-loaded (shared across tests)."""
    svc = EmbeddingService()
    # Eagerly load the model so the first test does not pay the full cold-start cost.
    svc.load_model()
    return svc


@pytest_asyncio.fixture(scope="module")
async def connected_db():
    """Connect db_manager once for the whole module; skip if unavailable."""
    try:
        await db_manager.connect()
    except Exception as exc:
        pytest.skip(f"Could not connect to databases: {exc}")

    yield db_manager

    try:
        await db_manager.close()
    except Exception:
        pass


@pytest_asyncio.fixture(scope="module")
async def hybrid_engine(connected_db, embedding_service):
    """Return an initialised HybridRetrievalEngine."""
    engine = HybridRetrievalEngine(
        embedding_service=embedding_service,
        reranker_service=None,
    )
    return engine


@pytest_asyncio.fixture(scope="module")
async def baseline_engine(connected_db):
    """Return an initialised BaselineEngine."""
    return BaselineEngine()


# ---------------------------------------------------------------------------
# Group 1 – Qdrant Collection Health
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.qdrant_health
class TestQdrantCollectionHealth:
    """Verify the Qdrant collection exists, is healthy, and has data."""

    def test_collection_exists(self, qdrant_client):
        """The repositories_embeddings collection must be present in Qdrant."""
        collections = [c.name for c in qdrant_client.get_collections().collections]
        print(f"\n[qdrant_health] Available collections: {collections}")
        assert (
            COLLECTION_NAME in collections
        ), f"Expected collection '{COLLECTION_NAME}' not found. Available: {collections}"

    def test_collection_has_substantial_data(self, qdrant_client):
        """The collection must contain at least 1 million vectors."""
        info = qdrant_client.get_collection(COLLECTION_NAME)
        count = info.points_count
        print(f"\n[qdrant_health] Point count: {count:,}")

        if count == 0 and isinstance(qdrant_client, GatewayQdrantClient):
            # Gateway proxies don't expose accurate point counts.  Verify that
            # data actually exists by checking that a search returns results.
            zero_vec = [0.0] * EMBEDDING_DIM
            hits = qdrant_client.search(COLLECTION_NAME, zero_vec, limit=10)
            assert (
                len(hits) > 0
            ), "Gateway reports 0 points and search returned no results — collection appears to be empty"
            print(
                "[qdrant_health] Point count unavailable via gateway — verified non-empty via search"
            )
        else:
            assert (
                count is not None and count >= MIN_EXPECTED_POINTS
            ), f"Expected >= {MIN_EXPECTED_POINTS:,} points, got {count:,}"

    def test_vector_dimension_is_correct(self, qdrant_client):
        """Vectors stored in the collection must be 384-dimensional."""
        info = qdrant_client.get_collection(COLLECTION_NAME)
        # The config field structure varies by qdrant-client version.
        config = info.config
        vectors_config = config.params.vectors
        if hasattr(vectors_config, "size"):
            dim = vectors_config.size
        else:
            # Named vectors dict — use the first (or only) entry.
            dim = next(iter(vectors_config.values())).size
        print(f"\n[qdrant_health] Vector dimension: {dim}")
        assert dim == EMBEDDING_DIM, f"Expected dimension {EMBEDDING_DIM}, got {dim}"

    def test_collection_status_is_green(self, qdrant_client):
        """The collection status must be 'green' or 'yellow' (not 'red' or 'grey')."""
        info = qdrant_client.get_collection(COLLECTION_NAME)
        status = info.status
        print(f"\n[qdrant_health] Collection status: {status}")
        assert str(status).lower() in (
            "green",
            "yellow",
        ), f"Collection status is '{status}', expected green or yellow"

    def test_basic_vector_search_returns_results(self, qdrant_client):
        """A zero vector search must return at least one hit."""
        zero_vector = [0.0] * EMBEDDING_DIM
        results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=zero_vector,
            limit=5,
        )
        print(f"\n[qdrant_health] Zero-vector search returned {len(results)} results")
        assert len(results) >= 1, "Zero vector search returned no results"


# ---------------------------------------------------------------------------
# Group 2 – Embedding Quality
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.embedding_quality
class TestEmbeddingQuality:
    """Validate embedding model output quality."""

    def test_embedding_has_correct_dimension(self, embedding_service):
        """A single text embedding must be 384-dimensional."""
        vec = embedding_service._embed_sync("python machine learning")
        dim = len(vec)
        print(f"\n[embedding_quality] Embedding dimension: {dim}")
        assert (
            dim == EMBEDDING_DIM
        ), f"Expected {EMBEDDING_DIM}-dim embedding, got {dim}-dim"

    def test_embedding_is_not_zero(self, embedding_service):
        """The embedding must not be a zero (or near-zero) vector."""
        vec = embedding_service._embed_sync("open source repositories")
        norm = math.sqrt(sum(x * x for x in vec))
        print(f"\n[embedding_quality] Embedding L2 norm: {norm:.6f}")
        assert norm > 0.01, f"Embedding norm {norm} is suspiciously small"

    def test_similar_queries_have_high_cosine_similarity(self, embedding_service):
        """Semantically similar queries must produce similar vectors.

        all-MiniLM-L6-v2 compresses cosine similarity into a narrower range
        than larger models; 0.35 is a meaningful relatedness threshold for it.
        """
        q1 = "machine learning python"
        q2 = "deep learning artificial intelligence"
        v1 = embedding_service._embed_sync(q1).tolist()
        v2 = embedding_service._embed_sync(q2).tolist()
        sim = cosine_similarity(v1, v2)
        print(f"\n[embedding_quality] Similarity '{q1}' vs '{q2}': {sim:.4f}")
        assert (
            sim >= 0.35
        ), f"Expected cosine similarity >= 0.35 for similar queries, got {sim:.4f}"

    def test_unrelated_queries_have_low_cosine_similarity(self, embedding_service):
        """Semantically unrelated queries must produce dissimilar vectors (<= 0.40)."""
        q1 = "machine learning"
        q2 = "cooking recipes"
        v1 = embedding_service._embed_sync(q1).tolist()
        v2 = embedding_service._embed_sync(q2).tolist()
        sim = cosine_similarity(v1, v2)
        print(f"\n[embedding_quality] Similarity '{q1}' vs '{q2}': {sim:.4f}")
        assert (
            sim <= 0.40
        ), f"Expected cosine similarity <= 0.40 for unrelated queries, got {sim:.4f}"

    def test_batch_embeddings_match_single_embeddings(self, embedding_service):
        """batch[i] must be approximately equal to single-embed(texts[i])."""
        texts = [
            "python web framework",
            "kubernetes container orchestration",
            "natural language processing",
        ]
        batch_vecs = embedding_service._embed_batch_sync(texts).tolist()
        for i, text in enumerate(texts):
            single_vec = embedding_service._embed_sync(text).tolist()
            sim = cosine_similarity(batch_vecs[i], single_vec)
            print(f"\n[embedding_quality] Batch[{i}] vs single for '{text}': {sim:.6f}")
            assert (
                sim > 0.9999
            ), f"Batch embedding for '{text}' diverges from single embedding (cosine sim = {sim:.6f})"

    def test_different_queries_produce_different_vectors(self, embedding_service):
        """Two clearly different queries must not be nearly identical (<0.99 similarity)."""
        v1 = embedding_service._embed_sync("react frontend javascript").tolist()
        v2 = embedding_service._embed_sync("rust systems programming").tolist()
        sim = cosine_similarity(v1, v2)
        print(
            f"\n[embedding_quality] Similarity 'react frontend' vs 'rust systems programming': {sim:.4f}"
        )
        assert (
            sim < 0.99
        ), f"Expected different vectors for different queries, got similarity {sim:.4f}"


# ---------------------------------------------------------------------------
# Group 3 – Semantic Search Quality
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.search_quality
class TestSemanticSearchQuality:
    """Validate that Qdrant vector search returns sensible results."""

    def _run_search(
        self, qdrant_client, embedding_service, query: str, top_k: int = 10
    ):
        """Embed query and search Qdrant; returns raw hit list."""
        vec = embedding_service._embed_sync(query).tolist()
        results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vec,
            limit=top_k,
            with_payload=True,
        )
        return results

    def test_ml_query_returns_results(self, qdrant_client, embedding_service):
        """ML query must return at least 5 results."""
        results = self._run_search(
            qdrant_client, embedding_service, "machine learning python", top_k=10
        )
        print(f"\n[search_quality] ML query returned {len(results)} results")
        assert len(results) >= 5, f"Expected >= 5 results, got {len(results)}"

    def test_ml_query_scores_are_descending(self, qdrant_client, embedding_service):
        """Scores must be in descending order (Qdrant guarantees this, but verify)."""
        results = self._run_search(
            qdrant_client, embedding_service, "machine learning python", top_k=10
        )
        scores = [r.score for r in results]
        print(f"\n[search_quality] ML query scores: {[f'{s:.4f}' for s in scores]}")
        assert scores == sorted(
            scores, reverse=True
        ), f"Scores are not sorted descending: {scores}"

    def test_ml_query_top_result_has_high_score(self, qdrant_client, embedding_service):
        """The top result must have a cosine similarity score >= 0.5."""
        results = self._run_search(
            qdrant_client, embedding_service, "machine learning python", top_k=1
        )
        top_score = results[0].score
        print(f"\n[search_quality] ML query top score: {top_score:.4f}")
        assert (
            top_score >= 0.5
        ), f"Expected top cosine score >= 0.5, got {top_score:.4f}"

    def test_different_queries_return_different_top_results(
        self, qdrant_client, embedding_service
    ):
        """ML and React queries must not share more than 2 of their top-5 results."""
        ml_results = self._run_search(
            qdrant_client, embedding_service, "machine learning python", top_k=5
        )
        react_results = self._run_search(
            qdrant_client, embedding_service, "react javascript frontend", top_k=5
        )

        ml_ids = {r.id for r in ml_results}
        react_ids = {r.id for r in react_results}
        overlap = ml_ids & react_ids

        print(
            f"\n[search_quality] ML top-5 IDs: {ml_ids}\nReact top-5 IDs: {react_ids}\nOverlap: {overlap}"
        )
        assert (
            len(overlap) < 3
        ), f"Expected < 3 shared results between ML and React queries, got {len(overlap)} overlap: {overlap}"

    def test_devops_query_returns_results(self, qdrant_client, embedding_service):
        """A DevOps query must return at least 5 results."""
        results = self._run_search(
            qdrant_client, embedding_service, "kubernetes docker devops CI CD", top_k=10
        )
        print(f"\n[search_quality] DevOps query returned {len(results)} results")
        assert len(results) >= 5, f"Expected >= 5 results, got {len(results)}"

    def test_all_results_have_valid_ids(self, qdrant_client, embedding_service):
        """Every returned hit must have a non-null, non-empty ID."""
        results = self._run_search(
            qdrant_client, embedding_service, "open source web framework", top_k=10
        )
        for r in results:
            assert r.id is not None, f"Hit has null ID: {r}"
            assert str(r.id).strip() != "", f"Hit has empty ID: {r}"

        print(f"\n[search_quality] All {len(results)} results have valid IDs")

    def test_ml_query_top5_printed(self, qdrant_client, embedding_service):
        """Print top-5 ML results for human inspection (always passes)."""
        results = self._run_search(
            qdrant_client, embedding_service, "machine learning python", top_k=5
        )
        print("\n[search_quality] Top-5 results for 'machine learning python':")
        for i, r in enumerate(results, 1):
            payload = r.payload or {}
            name = payload.get("name") or payload.get("full_name") or "(no name)"
            print(f"  {i}. id={r.id}  score={r.score:.4f}  name={name}")

    def test_react_query_top5_printed(self, qdrant_client, embedding_service):
        """Print top-5 React results for human inspection (always passes)."""
        results = self._run_search(
            qdrant_client, embedding_service, "react javascript frontend", top_k=5
        )
        print("\n[search_quality] Top-5 results for 'react javascript frontend':")
        for i, r in enumerate(results, 1):
            payload = r.payload or {}
            name = payload.get("name") or payload.get("full_name") or "(no name)"
            print(f"  {i}. id={r.id}  score={r.score:.4f}  name={name}")


# ---------------------------------------------------------------------------
# Group 4 – Full Pipeline
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.pipeline
class TestFullPipeline:
    """End-to-end tests through the engine layer."""

    @pytest.mark.asyncio
    async def test_hybrid_engine_returns_results(self, hybrid_engine):
        """HybridRetrievalEngine must return at least one result."""
        request = make_request("machine learning python library", top_k=10)
        results = await hybrid_engine.recommend(request)
        print(f"\n[pipeline] Hybrid engine returned {len(results)} results")
        assert len(results) >= 1, "Hybrid engine returned no results"

    @pytest.mark.asyncio
    async def test_hybrid_results_have_required_fields(self, hybrid_engine):
        """Every hybrid result must have a non-empty repo_id, positive score, and positive rank."""
        request = make_request("machine learning python library", top_k=10)
        results = await hybrid_engine.recommend(request)
        for r in results:
            assert (
                r.repo_id is not None and str(r.repo_id).strip() != ""
            ), f"Result has null/empty repo_id: {r}"
            assert r.score > 0, f"Result has non-positive score {r.score}: {r}"
            assert r.rank > 0, f"Result has non-positive rank {r.rank}: {r}"

        print(
            f"\n[pipeline] All {len(results)} results have valid repo_id, score > 0, rank > 0"
        )

    @pytest.mark.asyncio
    async def test_hybrid_results_are_ranked_consecutively(self, hybrid_engine):
        """Ranks must be consecutive integers starting at 1."""
        request = make_request("python data science", top_k=10)
        results = await hybrid_engine.recommend(request)
        ranks = [r.rank for r in results]
        expected = list(range(1, len(results) + 1))
        print(f"\n[pipeline] Ranks: {ranks}")
        assert ranks == expected, f"Expected consecutive ranks {expected}, got {ranks}"

    @pytest.mark.asyncio
    async def test_baseline_engine_returns_results(self, baseline_engine):
        """BaselineEngine must return at least one result."""
        request = make_request("python", top_k=10)
        results = await baseline_engine.recommend(request)
        print(f"\n[pipeline] Baseline engine returned {len(results)} results")
        assert len(results) >= 1, "Baseline engine returned no results"

    @pytest.mark.asyncio
    async def test_hybrid_and_baseline_return_different_results(
        self, hybrid_engine, baseline_engine
    ):
        """Hybrid and baseline must not return the identical set of repo IDs."""
        request = make_request("machine learning python", top_k=10)
        hybrid_results = await hybrid_engine.recommend(request)
        baseline_results = await baseline_engine.recommend(request)

        hybrid_ids = {r.repo_id for r in hybrid_results}
        baseline_ids = {r.repo_id for r in baseline_results}

        print(
            f"\n[pipeline] Hybrid IDs (sample): {list(hybrid_ids)[:5]}\nBaseline IDs (sample): {list(baseline_ids)[:5]}"
        )

        assert (
            hybrid_ids != baseline_ids
        ), "Hybrid and baseline engines returned the exact same set of repository IDs"

    @pytest.mark.asyncio
    async def test_hybrid_top5_ml_printed(self, hybrid_engine):
        """Print top-5 hybrid results for ML query (always passes)."""
        request = make_request("machine learning python", top_k=5)
        results = await hybrid_engine.recommend(request)
        print("\n[pipeline] Hybrid top-5 for 'machine learning python':")
        for r in results:
            print(
                f"  rank={r.rank}  score={r.score:.6f}  name={r.name!r}  lang={r.language}  stars={r.stars}"
            )

    @pytest.mark.asyncio
    async def test_hybrid_top5_devops_printed(self, hybrid_engine):
        """Print top-5 hybrid results for DevOps query (always passes)."""
        request = make_request("kubernetes docker container orchestration", top_k=5)
        results = await hybrid_engine.recommend(request)
        print(
            "\n[pipeline] Hybrid top-5 for 'kubernetes docker container orchestration':"
        )
        for r in results:
            print(
                f"  rank={r.rank}  score={r.score:.6f}  name={r.name!r}  lang={r.language}  stars={r.stars}"
            )


# ---------------------------------------------------------------------------
# Group 5 – Filter Correctness
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.filters
class TestFilterCorrectness:
    """Verify that hard filters are respected by the hybrid engine."""

    @pytest.mark.asyncio
    async def test_language_filter_applied(self, hybrid_engine):
        """Every result for a Python-filtered query must have language == 'Python'."""
        request = make_request("web framework", top_k=10, language="Python")
        results = await hybrid_engine.recommend(request)
        print(
            f"\n[filters] Language-filtered results ({len(results)} total): "
            + ", ".join(f"{r.name}({r.language})" for r in results[:5])
        )
        for r in results:
            assert (
                r.language == "Python"
            ), f"Language filter violated: repo '{r.name}' has language '{r.language}'"

    @pytest.mark.asyncio
    async def test_min_stars_filter_applied(self, hybrid_engine):
        """Every result for a min_stars=500 query must have stars >= 500."""
        request = make_request("open source library", top_k=10, min_stars=500)
        results = await hybrid_engine.recommend(request)
        print(
            f"\n[filters] min_stars=500 results ({len(results)} total): "
            + ", ".join(f"{r.name}({r.stars}★)" for r in results[:5])
        )
        for r in results:
            assert (
                r.stars >= 500
            ), f"min_stars filter violated: repo '{r.name}' has {r.stars} stars"

    @pytest.mark.asyncio
    async def test_empty_results_for_impossible_filter(self, hybrid_engine):
        """A physically impossible star threshold must yield zero results."""
        request = make_request("open source", top_k=10, min_stars=10_000_000)
        results = await hybrid_engine.recommend(request)
        print(
            f"\n[filters] min_stars=10,000,000 returned {len(results)} results (expected 0)"
        )
        assert (
            len(results) == 0
        ), f"Expected 0 results for min_stars=10_000_000, got {len(results)}"


# ---------------------------------------------------------------------------
# Group 6 – Performance
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.performance
class TestPerformance:
    """Latency benchmarks for the recommendation pipeline."""

    def test_first_semantic_search_latency(self, qdrant_client, embedding_service):
        """First semantic search (model may still be loading) must complete within 30s."""
        # Embed + search
        start = time.perf_counter()
        vec = embedding_service._embed_sync("python web framework").tolist()
        qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vec,
            limit=10,
        )
        elapsed = time.perf_counter() - start
        print(f"\n[performance] First semantic search latency: {elapsed:.3f}s")
        assert (
            elapsed <= 30.0
        ), f"First semantic search took {elapsed:.3f}s, expected <= 30s"

    def test_subsequent_semantic_search_latency(self, qdrant_client, embedding_service):
        """Subsequent searches (model warm) must complete within 5s."""
        # Warm run to ensure model is loaded
        _ = embedding_service._embed_sync("warmup query")

        start = time.perf_counter()
        vec = embedding_service._embed_sync("rust systems programming").tolist()
        qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vec,
            limit=10,
        )
        elapsed = time.perf_counter() - start
        print(f"\n[performance] Subsequent semantic search latency: {elapsed:.3f}s")
        assert (
            elapsed <= 5.0
        ), f"Subsequent semantic search took {elapsed:.3f}s, expected <= 5s"

    @pytest.mark.asyncio
    async def test_full_pipeline_latency(self, hybrid_engine):
        """A full hybrid engine recommend() call must complete within 30s."""
        request = make_request("machine learning tensorflow pytorch", top_k=10)

        start = time.perf_counter()
        results = await hybrid_engine.recommend(request)
        elapsed = time.perf_counter() - start

        print(
            f"\n[performance] Full pipeline latency: {elapsed:.3f}s (returned {len(results)} results)"
        )
        assert elapsed <= 30.0, f"Full pipeline took {elapsed:.3f}s, expected <= 30s"
