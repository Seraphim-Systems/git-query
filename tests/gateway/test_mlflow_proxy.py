from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.gateway.routers.mlflow_proxy as mlflow_proxy


class DummyAsyncClient:
    requests = []
    fail_hosts = set()
    status_by_prefix = {}

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def request(self, method, url, headers=None, content=None):
        self.__class__.requests.append(
            {
                "method": method,
                "url": str(url),
                "headers": headers or {},
                "content": content,
            }
        )
        for blocked_host in self.__class__.fail_hosts:
            if str(url).startswith(blocked_host):
                raise mlflow_proxy.httpx.ConnectError(
                    "connect failed",
                    request=mlflow_proxy.httpx.Request(method=method, url=url),
                )

        status_code = 200
        for prefix, override_status in self.__class__.status_by_prefix.items():
            if str(url).startswith(prefix):
                status_code = override_status
                break

        return mlflow_proxy.httpx.Response(
            status_code=status_code,
            content=b"ok",
            headers={"content-type": "text/plain"},
        )


def _build_test_app():
    app = FastAPI()
    app.include_router(mlflow_proxy.router, prefix="/mlflow")
    app.include_router(mlflow_proxy.static_router)
    return app


async def _allow_admin(_request):
    return "admin-user"


def test_proxy_preserves_mlflow_prefix(monkeypatch):
    monkeypatch.setattr(mlflow_proxy, "require_admin", _allow_admin)
    monkeypatch.setattr(mlflow_proxy.httpx, "AsyncClient", DummyAsyncClient)
    monkeypatch.setattr(
        mlflow_proxy,
        "MLFLOW_INTERNAL_URLS",
        ("http://git-query-mlflow:5000",),
    )

    DummyAsyncClient.requests = []
    DummyAsyncClient.fail_hosts = set()
    DummyAsyncClient.status_by_prefix = {}

    app = _build_test_app()
    client = TestClient(app)

    response = client.get("/mlflow/")

    assert response.status_code == 200
    assert DummyAsyncClient.requests[0]["url"] == "http://git-query-mlflow:5000/mlflow"


def test_proxy_falls_back_to_second_upstream_on_connect_error(monkeypatch):
    monkeypatch.setattr(mlflow_proxy, "require_admin", _allow_admin)
    monkeypatch.setattr(mlflow_proxy.httpx, "AsyncClient", DummyAsyncClient)
    monkeypatch.setattr(
        mlflow_proxy,
        "MLFLOW_INTERNAL_URLS",
        ("http://git-query-mlflow:5000", "http://mlflow:5000"),
    )

    DummyAsyncClient.requests = []
    DummyAsyncClient.fail_hosts = {"http://git-query-mlflow:5000"}
    DummyAsyncClient.status_by_prefix = {}

    app = _build_test_app()
    client = TestClient(app)

    response = client.get("/mlflow/api/2.0/mlflow/experiments/list")

    assert response.status_code == 200
    assert DummyAsyncClient.requests[0]["url"].startswith(
        "http://git-query-mlflow:5000/"
    )
    assert DummyAsyncClient.requests[1]["url"].startswith("http://mlflow:5000/")
    assert (
        DummyAsyncClient.requests[1]["url"]
        == "http://mlflow:5000/mlflow/api/2.0/mlflow/experiments/list"
    )


def test_proxy_retries_without_prefix_after_prefixed_404(monkeypatch):
    monkeypatch.setattr(mlflow_proxy, "require_admin", _allow_admin)
    monkeypatch.setattr(mlflow_proxy.httpx, "AsyncClient", DummyAsyncClient)
    monkeypatch.setattr(
        mlflow_proxy,
        "MLFLOW_INTERNAL_URLS",
        ("http://git-query-mlflow:5000",),
    )

    DummyAsyncClient.requests = []
    DummyAsyncClient.fail_hosts = set()
    DummyAsyncClient.status_by_prefix = {"http://git-query-mlflow:5000/mlflow": 404}

    app = _build_test_app()
    client = TestClient(app)

    response = client.get("/mlflow/")

    assert response.status_code == 200
    assert DummyAsyncClient.requests[0]["url"] == "http://git-query-mlflow:5000/mlflow"
    assert DummyAsyncClient.requests[1]["url"] == "http://git-query-mlflow:5000/"


def test_proxy_retries_static_files_root_after_prefixed_and_plain_404(monkeypatch):
    monkeypatch.setattr(mlflow_proxy, "require_admin", _allow_admin)
    monkeypatch.setattr(mlflow_proxy.httpx, "AsyncClient", DummyAsyncClient)
    monkeypatch.setattr(
        mlflow_proxy,
        "MLFLOW_INTERNAL_URLS",
        ("http://git-query-mlflow:5000",),
    )

    DummyAsyncClient.requests = []
    DummyAsyncClient.fail_hosts = set()
    DummyAsyncClient.status_by_prefix = {
        "http://git-query-mlflow:5000/mlflow": 404,
        "http://git-query-mlflow:5000/": 404,
    }

    app = _build_test_app()
    client = TestClient(app)

    response = client.get("/mlflow/")

    assert response.status_code == 200
    assert DummyAsyncClient.requests[0]["url"] == "http://git-query-mlflow:5000/mlflow"
    assert DummyAsyncClient.requests[1]["url"] == "http://git-query-mlflow:5000/"
    assert (
        DummyAsyncClient.requests[2]["url"]
        == "http://git-query-mlflow:5000/mlflow/static-files"
    )


def test_static_files_proxy_falls_back_to_prefixed_static_path(monkeypatch):
    monkeypatch.setattr(mlflow_proxy, "require_admin", _allow_admin)
    monkeypatch.setattr(mlflow_proxy.httpx, "AsyncClient", DummyAsyncClient)
    monkeypatch.setattr(
        mlflow_proxy,
        "MLFLOW_INTERNAL_URLS",
        ("http://git-query-mlflow:5000",),
    )

    DummyAsyncClient.requests = []
    DummyAsyncClient.fail_hosts = set()
    DummyAsyncClient.status_by_prefix = {
        "http://git-query-mlflow:5000/static-files": 404,
    }

    app = _build_test_app()
    client = TestClient(app)

    response = client.get("/static-files/static/js/main.js")

    assert response.status_code == 200
    assert (
        DummyAsyncClient.requests[0]["url"]
        == "http://git-query-mlflow:5000/static-files/static/js/main.js"
    )
    assert (
        DummyAsyncClient.requests[1]["url"]
        == "http://git-query-mlflow:5000/mlflow/static-files/static/js/main.js"
    )
