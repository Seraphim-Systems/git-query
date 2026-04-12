"""Docker end-to-end test for the recommender service.

Builds the Dockerfile.recommender image, starts the container with live
credentials from .env, waits for /health to be ready, fires real HTTP
requests, asserts quality, then tears down cleanly.

Run with:
    pytest tests/test_docker_e2e.py -m e2e -v -s

Requirements:
    - Docker daemon running locally
    - .env file with QDRANT_URL, QDRANT_API_KEY (or APIKEY_QDRANT),
      MONGODB_URL, REDIS_URL populated
    - ~5-10 minutes for first run (image build + model download)
"""

import os
import time
import subprocess
import json
import pytest
import requests
from dotenv import dotenv_values

# ── Constants ─────────────────────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCKERFILE = "infrastructure/docker/Dockerfile.recommender"
IMAGE_TAG = "git-query-recommender-test"
CONTAINER_NAME = "git-query-recommender-e2e"
HOST_PORT = 18095  # Use a non-standard port to avoid conflicts
CONTAINER_PORT = 8095
BASE_URL = f"http://localhost:{HOST_PORT}"
HEALTH_TIMEOUT_S = 120  # Model download + startup
HEALTH_POLL_INTERVAL_S = 3

# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_env_for_container() -> list[str]:
    """Load .env and return as list of -e KEY=VALUE strings for docker run."""
    env_path = os.path.join(PROJECT_ROOT, ".env")
    if not os.path.exists(env_path):
        return []

    values = dotenv_values(env_path)
    # Pass only the vars the recommender needs
    relevant_keys = {
        "QDRANT_URL",
        "QDRANT_API_KEY",
        "APIKEY_QDRANT",
        "QDRANT_HOST",
        "QDRANT_HTTP_PORT",
        "QDRANT_COLLECTION",
        "MONGODB_URL",
        "REDIS_URL",
        "EMBEDDING_API_KEY",
        "LOG_LEVEL",
    }
    args = []
    for key in relevant_keys:
        if key in values and values[key]:
            args += ["-e", f"{key}={values[key]}"]
    return args


def _docker_available() -> bool:
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _container_running(name: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Running}}", name], capture_output=True, text=True
    )
    return result.stdout.strip() == "true"


def _stop_and_remove_container(name: str):
    subprocess.run(["docker", "stop", name], capture_output=True)
    subprocess.run(["docker", "rm", name], capture_output=True)


def _wait_for_health(url: str, timeout: int, interval: int) -> bool:
    """Poll /health until 200 or timeout. Returns True if healthy."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(f"{url}/health", timeout=5)
            if resp.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        print(f"  Waiting for container... ({int(deadline - time.time())}s remaining)")
        time.sleep(interval)
    return False


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def docker_available():
    if not _docker_available():
        pytest.skip("Docker daemon not available")


@pytest.fixture(scope="module")
def built_image(docker_available):
    """Build the recommender Docker image. Cached by Docker layer cache."""
    print(f"\nBuilding Docker image: {IMAGE_TAG}")
    result = subprocess.run(
        [
            "docker",
            "build",
            "-f",
            DOCKERFILE,
            "-t",
            IMAGE_TAG,
            ".",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=600,  # 10 min — first build downloads model weights
    )
    if result.returncode != 0:
        pytest.fail(f"Docker build failed:\n{result.stdout[-2000:]}\n{result.stderr[-2000:]}")
    print(f"Image built: {IMAGE_TAG}")
    return IMAGE_TAG


@pytest.fixture(scope="module")
def running_container(built_image):
    """Start the container, wait for health, yield base URL, then tear down."""
    # Clean up any leftover container from a previous run
    _stop_and_remove_container(CONTAINER_NAME)

    env_args = _load_env_for_container()
    if not env_args:
        pytest.skip("No .env found — cannot pass credentials to container")

    cmd = [
        "docker",
        "run",
        "--name",
        CONTAINER_NAME,
        "-d",
        "-p",
        f"{HOST_PORT}:{CONTAINER_PORT}",
        *env_args,
        built_image,
    ]

    print(f"\nStarting container: {CONTAINER_NAME}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"docker run failed:\n{result.stderr}")

    print(f"Waiting up to {HEALTH_TIMEOUT_S}s for /health...")
    healthy = _wait_for_health(BASE_URL, HEALTH_TIMEOUT_S, HEALTH_POLL_INTERVAL_S)

    if not healthy:
        # Capture logs for debugging
        logs = subprocess.run(["docker", "logs", "--tail", "50", CONTAINER_NAME], capture_output=True, text=True).stdout
        _stop_and_remove_container(CONTAINER_NAME)
        pytest.fail(f"Container did not become healthy within {HEALTH_TIMEOUT_S}s.\nLast logs:\n{logs}")

    print(f"Container healthy at {BASE_URL}")
    yield BASE_URL

    # Teardown
    print(f"\nStopping container: {CONTAINER_NAME}")
    _stop_and_remove_container(CONTAINER_NAME)


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.e2e
class TestDockerE2E:
    """End-to-end tests through the built Docker container."""

    def test_health_endpoint(self, running_container):
        """Container reports healthy after startup."""
        resp = requests.get(f"{running_container}/health", timeout=10)
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "healthy"
        assert body["service"] == "recommender"
        assert "version" in body
        print(f"\n[health] {body}")

    def test_recommend_returns_results(self, running_container):
        """POST /recommend returns a valid response with results."""
        payload = {"query": "machine learning python", "top_k": 5}
        resp = requests.post(
            f"{running_container}/recommend",
            json=payload,
            timeout=60,  # First request loads model
        )
        assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text[:500]}"
        body = resp.json()
        assert "results" in body
        assert "processing_time_ms" in body
        assert body["processing_time_ms"] > 0

        print(f"\n[recommend] query='machine learning python' → {len(body['results'])} results")
        for r in body["results"][:5]:
            print(f"  {r['rank']}. {r['name']} (score={r['score']:.3f})")

    def test_recommend_different_queries_differ(self, running_container):
        """Different queries return different top results."""
        r1 = requests.post(
            f"{running_container}/recommend",
            json={"query": "machine learning neural network", "top_k": 5},
            timeout=30,
        ).json()
        r2 = requests.post(
            f"{running_container}/recommend",
            json={"query": "kubernetes docker orchestration", "top_k": 5},
            timeout=30,
        ).json()

        ids1 = {r["repo_id"] for r in r1.get("results", [])}
        ids2 = {r["repo_id"] for r in r2.get("results", [])}
        overlap = ids1 & ids2

        print(f"\n[diversity] ML overlap with DevOps: {len(overlap)}/5 repos shared")
        assert len(overlap) < 4, f"Too much overlap ({len(overlap)}/5) — search may not be query-sensitive"

    def test_recommend_language_filter(self, running_container):
        """language filter is enforced in the containerised service."""
        resp = requests.post(
            f"{running_container}/recommend",
            json={"query": "web framework", "top_k": 10, "language": "Python"},
            timeout=30,
        )
        assert resp.status_code == 200
        results = resp.json().get("results", [])
        non_python = [r for r in results if r.get("language") and r["language"] != "Python"]
        print(f"\n[filter] language=Python → {len(results)} results, {len(non_python)} non-Python")
        assert len(non_python) == 0, f"Filter leaked non-Python results: {[r['language'] for r in non_python]}"

    def test_recommend_min_stars_filter(self, running_container):
        """min_stars filter is enforced in the containerised service."""
        min_stars = 500
        resp = requests.post(
            f"{running_container}/recommend",
            json={"query": "machine learning", "top_k": 10, "min_stars": min_stars},
            timeout=30,
        )
        assert resp.status_code == 200
        results = resp.json().get("results", [])
        below = [r for r in results if r.get("stars", 0) < min_stars]
        print(f"\n[filter] min_stars={min_stars} → {len(results)} results, {len(below)} below threshold")
        assert len(below) == 0, f"Filter leaked low-star results: {[r['stars'] for r in below]}"

    def test_recommend_baseline_variant(self, running_container):
        """Baseline variant responds correctly."""
        resp = requests.post(
            f"{running_container}/recommend",
            json={"query": "python", "top_k": 5, "variant": "baseline"},
            timeout=30,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["variant"] == "baseline"
        print(f"\n[baseline] variant={body['variant']}, results={len(body['results'])}")

    def test_admin_engines_lists_three_engines(self, running_container):
        """All 3 engines are registered and listed."""
        resp = requests.get(f"{running_container}/admin/engines", timeout=10)
        assert resp.status_code == 200
        engines = resp.json().get("engines", [])
        names = [e.get("name") for e in engines]
        print(f"\n[engines] registered: {names}")
        assert "baseline" in names, f"baseline missing from engines: {names}"
        assert "hybrid" in names, f"hybrid missing from engines: {names}"
        assert "personalized" in names, f"personalized missing from engines: {names}"

    def test_log_interaction(self, running_container):
        """POST /interaction logs without error."""
        from datetime import datetime, timezone

        payload = {
            "user_id": "e2e-test-user",
            "query": "machine learning python",
            "repo_id": "e2e-test-repo-123",
            "interaction_type": "view",
            "variant": "baseline",
        }
        resp = requests.post(
            f"{running_container}/interaction",
            json=payload,
            timeout=10,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert "interaction_id" in body
        print(f"\n[interaction] logged: {body['interaction_id']}")

    def test_recommend_invalid_request_returns_422(self, running_container):
        """Missing required fields returns HTTP 422, not 500."""
        resp = requests.post(
            f"{running_container}/recommend",
            json={"top_k": 5},  # missing required 'query'
            timeout=10,
        )
        assert resp.status_code == 422, f"Expected 422 for invalid request, got {resp.status_code}"

    def test_response_time_is_acceptable(self, running_container):
        """Warm queries complete within 10 seconds."""
        # Fire a warm-up first
        requests.post(
            f"{running_container}/recommend",
            json={"query": "warm up", "top_k": 3},
            timeout=60,
        )

        start = time.time()
        resp = requests.post(
            f"{running_container}/recommend",
            json={"query": "web scraping python", "top_k": 10},
            timeout=30,
        )
        elapsed = time.time() - start

        assert resp.status_code == 200
        print(f"\n[perf] warm query latency: {elapsed:.2f}s")
        assert elapsed <= 10.0, f"Warm query took {elapsed:.2f}s — expected <= 10s"

    def test_container_logs_show_no_import_errors(self, running_container):
        """Container logs must not contain ImportError or ModuleNotFoundError."""
        logs = (
            subprocess.run(["docker", "logs", CONTAINER_NAME], capture_output=True, text=True).stdout
            + subprocess.run(["docker", "logs", CONTAINER_NAME], capture_output=True, text=True).stderr
        )

        assert "ImportError" not in logs, "Container logs contain ImportError — check Dockerfile COPY paths"
        assert "ModuleNotFoundError" not in logs, (
            "Container logs contain ModuleNotFoundError — check Dockerfile COPY paths"
        )
        print(f"\n[logs] No import errors detected in container logs")
