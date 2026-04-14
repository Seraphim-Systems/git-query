from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.gateway.routers.mlflow_proxy as mlflow_proxy


class DummyAsyncClient:
    requests = []
    fail_hosts = set()

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
        return mlflow_proxy.httpx.Response(
            status_code=200,
            content=b"ok",
            headers={"content-type": "text/plain"},
        )


def _build_test_app():
    app = FastAPI()
    app.include_router(mlflow_proxy.router, prefix="/mlflow")
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
