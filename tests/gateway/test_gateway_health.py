import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.gateway.routers.health as health_router


class DummyMongoOK:
    async def command(self, cmd):
        assert cmd == "ping"
        return {"ok": 1}


class DummyMongoFail:
    async def command(self, cmd):
        raise RuntimeError("mongo down")


class DummyRedisOK:
    async def ping(self):
        return True


class DummyRedisFail:
    async def ping(self):
        raise RuntimeError("redis down")


class DummyQdrantCollections:
    def __init__(self, count):
        self.collections = [object() for _ in range(count)]


class DummyQdrantClientOK:
    def get_collections(self):
        return DummyQdrantCollections(3)


class DummyQdrantClientFail:
    def get_collections(self):
        raise RuntimeError("qdrant client failed")


class DummyHTTPResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


class DummyAsyncClient:
    next_get_response = DummyHTTPResponse(200)
    raise_on_get = None
    requested_urls = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, url):
        self.__class__.requested_urls.append(url)
        if self.__class__.raise_on_get:
            raise self.__class__.raise_on_get
        return self.__class__.next_get_response


@pytest.fixture
def health_app():
    app = FastAPI()
    app.include_router(health_router.router)
    return app


@pytest.fixture
def client(health_app):
    return TestClient(health_app)


def test_health_mongodb_ok(client):
    client.app.state.mongodb = DummyMongoOK()

    response = client.get("/api/health/mongodb")

    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "mongodb"
    assert body["result"]["status"] is True


def test_health_mongodb_unavailable_when_client_missing(client):
    response = client.get("/api/health/mongodb")

    assert response.status_code == 503
    body = response.json()
    assert body["service"] == "mongodb"
    assert body["result"]["status"] is False
    assert "mongodb client not available" in body["result"]["error"]


def test_health_mongodb_returns_503_on_exception(client):
    client.app.state.mongodb = DummyMongoFail()

    response = client.get("/api/health/mongodb")

    assert response.status_code == 503
    body = response.json()
    assert body["result"]["status"] is False
    assert body["result"]["error"] == "mongodb health check failed"


def test_health_redis_ok(client):
    client.app.state.redis = DummyRedisOK()

    response = client.get("/api/health/redis")

    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "redis"
    assert body["result"]["status"] is True


def test_health_redis_unavailable_when_client_missing(client):
    response = client.get("/api/health/redis")

    assert response.status_code == 503
    body = response.json()
    assert body["service"] == "redis"
    assert body["result"]["status"] is False
    assert "redis client not available" in body["result"]["error"]


def test_health_redis_returns_503_on_exception(client):
    client.app.state.redis = DummyRedisFail()

    response = client.get("/api/health/redis")

    assert response.status_code == 503
    body = response.json()
    assert body["result"]["status"] is False
    assert body["result"]["error"] == "redis health check failed"


def test_health_qdrant_ok_via_shared_client(client, monkeypatch):
    monkeypatch.setattr(
        health_router,
        "get_qdrant_client",
        lambda: DummyQdrantClientOK(),
    )

    response = client.get("/api/health/qdrant")

    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "qdrant"
    assert body["result"]["status"] is True
    assert body["result"]["collections"] == 3


def test_health_qdrant_falls_back_to_http_when_client_fails(client, monkeypatch):
    monkeypatch.setattr(
        health_router,
        "get_qdrant_client",
        lambda: DummyQdrantClientFail(),
    )
    DummyAsyncClient.next_get_response = DummyHTTPResponse(200)
    DummyAsyncClient.raise_on_get = None
    DummyAsyncClient.requested_urls = []

    monkeypatch.setattr(health_router.httpx, "AsyncClient", DummyAsyncClient)

    response = client.get("/api/health/qdrant")

    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "qdrant"
    assert body["result"]["status"] is True
    assert body["result"]["http_status"] == 200
    assert DummyAsyncClient.requested_urls


def test_health_qdrant_returns_503_when_http_fallback_fails(client, monkeypatch):
    monkeypatch.setattr(
        health_router,
        "get_qdrant_client",
        lambda: DummyQdrantClientFail(),
    )
    DummyAsyncClient.raise_on_get = RuntimeError("qdrant http down")
    DummyAsyncClient.requested_urls = []

    monkeypatch.setattr(health_router.httpx, "AsyncClient", DummyAsyncClient)

    response = client.get("/api/health/qdrant")

    assert response.status_code == 503
    body = response.json()
    assert body["result"]["status"] is False
    assert body["result"]["error"] == "qdrant health check failed"


def test_health_mcp_ok(client, monkeypatch):
    DummyAsyncClient.next_get_response = DummyHTTPResponse(200)
    DummyAsyncClient.raise_on_get = None
    DummyAsyncClient.requested_urls = []

    monkeypatch.setattr(health_router.httpx, "AsyncClient", DummyAsyncClient)

    response = client.get("/api/health/mcp")

    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "mcp"
    assert body["result"]["status"] is True
    assert DummyAsyncClient.requested_urls


def test_health_mcp_returns_503_on_exception(client, monkeypatch):
    DummyAsyncClient.raise_on_get = RuntimeError("mcp offline")
    DummyAsyncClient.requested_urls = []

    monkeypatch.setattr(health_router.httpx, "AsyncClient", DummyAsyncClient)

    response = client.get("/api/health/mcp")

    assert response.status_code == 503
    body = response.json()
    assert body["service"] == "mcp"
    assert body["result"]["status"] is False
    assert body["result"]["error"] == "mcp server health check failed"


def test_health_check_all_returns_healthy_when_any_service_is_up(client, monkeypatch):
    client.app.state.mongodb = DummyMongoOK()
    client.app.state.redis = DummyRedisFail()

    monkeypatch.setattr(
        health_router,
        "get_qdrant_client",
        lambda: DummyQdrantClientFail(),
    )
    DummyAsyncClient.next_get_response = DummyHTTPResponse(503)
    DummyAsyncClient.raise_on_get = None
    DummyAsyncClient.requested_urls = []

    monkeypatch.setattr(health_router.httpx, "AsyncClient", DummyAsyncClient)

    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["services"]["gateway"] is True
    assert body["services"]["mongodb"] is True
    assert body["services"]["redis"] is False
    assert body["services"]["qdrant"] is False
    assert body["services"]["mcp_server"] is False
    assert "timestamp" in body


def test_health_check_all_keeps_200_because_gateway_is_up(client, monkeypatch):
    client.app.state.mongodb = DummyMongoFail()
    client.app.state.redis = DummyRedisFail()

    monkeypatch.setattr(
        health_router,
        "get_qdrant_client",
        lambda: DummyQdrantClientFail(),
    )
    DummyAsyncClient.raise_on_get = RuntimeError("all down")
    DummyAsyncClient.requested_urls = []

    monkeypatch.setattr(health_router.httpx, "AsyncClient", DummyAsyncClient)

    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["services"]["gateway"] is True
    assert body["services"]["mongodb"] is False
    assert body["services"]["redis"] is False
    assert body["services"]["qdrant"] is False
    assert body["services"]["mcp_server"] is False


@pytest.mark.asyncio
async def test_health_check_databases_function_returns_healthy_if_any_db_is_up(monkeypatch):
    app = FastAPI()
    app.state.mongodb = DummyMongoOK()
    app.state.redis = DummyRedisFail()

    class DummyRequest:
        def __init__(self, app):
            self.app = app

    monkeypatch.setattr(
        health_router,
        "get_qdrant_client",
        lambda: DummyQdrantClientFail(),
    )
    DummyAsyncClient.raise_on_get = RuntimeError("qdrant down")
    DummyAsyncClient.requested_urls = []
    monkeypatch.setattr(health_router.httpx, "AsyncClient", DummyAsyncClient)

    response = await health_router.health_check_databases(DummyRequest(app))
    assert response.status_code == 200