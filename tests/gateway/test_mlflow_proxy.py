import gzip

from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.gateway.routers.mlflow_proxy as mlflow_proxy


class DummyAsyncClient:
    requests = []
    fail_hosts = set()
    status_by_prefix = {}
    content_by_prefix = {}
    content_type_by_prefix = {}
    response_headers_by_prefix = {}

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

        content = b"ok"
        for prefix, override_content in self.__class__.content_by_prefix.items():
            if str(url).startswith(prefix):
                content = override_content
                break

        content_type = "text/plain"
        for (
            prefix,
            override_content_type,
        ) in self.__class__.content_type_by_prefix.items():
            if str(url).startswith(prefix):
                content_type = override_content_type
                break

        response_headers = {"content-type": content_type}
        for (
            prefix,
            override_headers,
        ) in self.__class__.response_headers_by_prefix.items():
            if str(url).startswith(prefix):
                response_headers.update(override_headers)
                break

        return mlflow_proxy.httpx.Response(
            status_code=status_code,
            content=content,
            headers=response_headers,
        )


def _build_test_app():
    app = FastAPI()
    app.include_router(mlflow_proxy.router, prefix="/mlflow")
    app.include_router(mlflow_proxy.static_router)
    app.include_router(mlflow_proxy.ajax_router)
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
    DummyAsyncClient.content_by_prefix = {}
    DummyAsyncClient.content_type_by_prefix = {}
    DummyAsyncClient.response_headers_by_prefix = {}

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
    DummyAsyncClient.content_by_prefix = {}
    DummyAsyncClient.content_type_by_prefix = {}
    DummyAsyncClient.response_headers_by_prefix = {}

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


def test_proxy_returns_404_when_prefixed_route_missing(monkeypatch):
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
    DummyAsyncClient.content_by_prefix = {}
    DummyAsyncClient.content_type_by_prefix = {}
    DummyAsyncClient.response_headers_by_prefix = {}

    app = _build_test_app()
    client = TestClient(app)

    response = client.get("/mlflow/")

    assert response.status_code == 404
    assert DummyAsyncClient.requests[0]["url"] == "http://git-query-mlflow:5000/mlflow"
    assert len(DummyAsyncClient.requests) == 1


def test_proxy_root_does_not_fallback_to_legacy_paths(monkeypatch):
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
    DummyAsyncClient.content_by_prefix = {}
    DummyAsyncClient.content_type_by_prefix = {}
    DummyAsyncClient.response_headers_by_prefix = {}

    app = _build_test_app()
    client = TestClient(app)

    response = client.get("/mlflow/")

    assert response.status_code == 404
    assert DummyAsyncClient.requests[0]["url"] == "http://git-query-mlflow:5000/mlflow"
    assert len(DummyAsyncClient.requests) == 1


def test_static_files_proxy_uses_prefixed_static_path(monkeypatch):
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
    DummyAsyncClient.content_by_prefix = {}
    DummyAsyncClient.content_type_by_prefix = {}
    DummyAsyncClient.response_headers_by_prefix = {}

    app = _build_test_app()
    client = TestClient(app)

    response = client.get("/static-files/static/js/main.js")

    assert response.status_code == 200
    assert (
        DummyAsyncClient.requests[0]["url"]
        == "http://git-query-mlflow:5000/mlflow/static-files/static/js/main.js"
    )
    assert len(DummyAsyncClient.requests) == 1


def test_proxy_rewrites_html_static_files_to_mlflow_prefix(monkeypatch):
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
    DummyAsyncClient.content_by_prefix = {
        "http://git-query-mlflow:5000/mlflow": b'<script src="/static-files/static/js/main.js"></script>'
    }
    DummyAsyncClient.content_type_by_prefix = {
        "http://git-query-mlflow:5000/mlflow": "text/html"
    }
    DummyAsyncClient.response_headers_by_prefix = {}

    app = _build_test_app()
    client = TestClient(app)

    response = client.get("/mlflow/")

    assert response.status_code == 200
    assert "/mlflow/static-files/static/js/main.js" in response.content.decode("utf-8")


def test_proxy_strips_conditional_cache_headers_for_mlflow_html(monkeypatch):
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
    DummyAsyncClient.content_by_prefix = {
        "http://git-query-mlflow:5000/mlflow": b'<script src="/static-files/static/js/main.js"></script>'
    }
    DummyAsyncClient.content_type_by_prefix = {
        "http://git-query-mlflow:5000/mlflow": "text/html"
    }
    DummyAsyncClient.response_headers_by_prefix = {}

    app = _build_test_app()
    client = TestClient(app)

    response = client.get(
        "/mlflow/",
        headers={"If-None-Match": "abc123", "If-Modified-Since": "yesterday"},
    )

    assert response.status_code == 200
    forward_headers = {
        key.lower(): value
        for key, value in DummyAsyncClient.requests[0]["headers"].items()
    }
    assert "if-none-match" not in forward_headers
    assert "if-modified-since" not in forward_headers
    assert response.headers.get("cache-control") == "no-store, max-age=0"
    assert "/mlflow/static-files/static/js/main.js" in response.content.decode("utf-8")


def test_proxy_response_drops_representation_headers(monkeypatch):
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
    DummyAsyncClient.content_by_prefix = {
        "http://git-query-mlflow:5000/mlflow/static-files": b"console.log('ok');"
    }
    DummyAsyncClient.content_type_by_prefix = {
        "http://git-query-mlflow:5000/mlflow/static-files": "application/javascript"
    }
    DummyAsyncClient.response_headers_by_prefix = {
        "http://git-query-mlflow:5000/mlflow/static-files": {
            "content-encoding": "gzip",
            "content-length": "9",
        }
    }

    app = _build_test_app()
    client = TestClient(app)

    response = client.get("/static-files/static/js/main.js")

    assert response.status_code == 200
    assert response.headers.get("content-type", "").startswith("application/javascript")
    normalized_response_headers = {k.lower(): v for k, v in response.headers.items()}
    assert "content-encoding" not in normalized_response_headers

    normalized_forward_headers = {
        k.lower(): v for k, v in DummyAsyncClient.requests[0]["headers"].items()
    }
    assert normalized_forward_headers.get("accept-encoding") == "identity"


def test_ajax_api_proxy_routes_to_mlflow(monkeypatch):
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
    DummyAsyncClient.content_by_prefix = {}
    DummyAsyncClient.content_type_by_prefix = {}
    DummyAsyncClient.response_headers_by_prefix = {}

    app = _build_test_app()
    client = TestClient(app)

    response = client.get("/ajax-api/2.0/mlflow/experiments/search?max_results=20000")

    assert response.status_code == 200
    assert (
        DummyAsyncClient.requests[0]["url"]
        == "http://git-query-mlflow:5000/mlflow/ajax-api/2.0/mlflow/experiments/search?max_results=20000"
    )


def test_ajax_api_proxy_does_not_fallback_to_plain_route(monkeypatch):
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
        "http://git-query-mlflow:5000/mlflow/ajax-api": 404,
    }
    DummyAsyncClient.content_by_prefix = {}
    DummyAsyncClient.content_type_by_prefix = {}
    DummyAsyncClient.response_headers_by_prefix = {}

    app = _build_test_app()
    client = TestClient(app)

    response = client.get("/ajax-api/2.0/mlflow/experiments/search")

    assert response.status_code == 404
    assert (
        DummyAsyncClient.requests[0]["url"]
        == "http://git-query-mlflow:5000/mlflow/ajax-api/2.0/mlflow/experiments/search"
    )
    assert len(DummyAsyncClient.requests) == 1
