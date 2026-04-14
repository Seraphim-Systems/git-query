"""MLFlow reverse-proxy router.

Forwards all /mlflow/* requests to the MLFlow container on the internal
Docker network.  Admin session authentication is required.
"""

import logging
import os

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import Response

from src.gateway.middleware.session import require_admin

logger = logging.getLogger(__name__)


def _load_internal_urls() -> tuple[str, ...]:
    raw_urls = os.getenv(
        "MLFLOW_INTERNAL_URLS",
        "http://git-query-mlflow:5000,http://mlflow:5000",
    )
    parsed = tuple(
        url.strip().rstrip("/") for url in raw_urls.split(",") if url.strip()
    )
    if parsed:
        return parsed
    return ("http://git-query-mlflow:5000", "http://mlflow:5000")


MLFLOW_INTERNAL_URLS = _load_internal_urls()

# Hop-by-hop headers that must not be forwarded
_HOP_BY_HOP = frozenset(
    {
        "transfer-encoding",
        "connection",
        "keep-alive",
        "te",
        "trailers",
        "upgrade",
        "proxy-authenticate",
        "proxy-authorization",
    }
)

# Headers tied to upstream payload representation that can become invalid
# once httpx normalizes/decodes response content.
_UNSAFE_PAYLOAD_HEADERS = frozenset({"content-length", "content-encoding"})

_METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]

router = APIRouter(tags=["MLFlow"])
static_router = APIRouter(tags=["MLFlow"])

_CONDITIONAL_CACHE_HEADERS = frozenset({"if-none-match", "if-modified-since"})


def _rewrite_mlflow_html(content: bytes, content_type: str | None) -> bytes:
    """Ensure MLflow HTML references static assets under /mlflow for proxy safety."""
    if not content or not content_type or "text/html" not in content_type.lower():
        return content

    text = content.decode("utf-8", errors="ignore")
    rewritten = text.replace('"/static-files/', '"/mlflow/static-files/')
    rewritten = rewritten.replace("'/static-files/", "'/mlflow/static-files/")

    if rewritten == text:
        return content
    return rewritten.encode("utf-8")


def _build_targets(base_url: str, path: str, query: str) -> tuple[str, ...]:
    normalized_path = path.lstrip("/")
    prefixed_path = "/mlflow" if not normalized_path else f"/mlflow/{normalized_path}"
    plain_path = "/" if not normalized_path else f"/{normalized_path}"

    # Prefer prefixed routing first (for servers configured behind /mlflow),
    # then fall back to plain routing for older MLflow versions.
    candidate_paths = [prefixed_path]
    if plain_path != prefixed_path:
        candidate_paths.append(plain_path)

    # Compatibility fallback for older/misconfigured deployments where MLflow
    # UI root is served under /mlflow/static-files.
    if not normalized_path:
        candidate_paths.append("/mlflow/static-files")

    targets: list[str] = []
    for upstream_path in candidate_paths:
        target = f"{base_url}{upstream_path}"
        if query:
            target = f"{target}?{query}"
        targets.append(target)
    return tuple(targets)


def _build_static_targets(base_url: str, path: str, query: str) -> tuple[str, ...]:
    normalized_path = path.lstrip("/")
    direct_path = (
        "/static-files" if not normalized_path else f"/static-files/{normalized_path}"
    )
    prefixed_path = (
        "/mlflow/static-files"
        if not normalized_path
        else f"/mlflow/static-files/{normalized_path}"
    )

    candidate_paths = [direct_path]
    if prefixed_path != direct_path:
        candidate_paths.append(prefixed_path)

    targets: list[str] = []
    for upstream_path in candidate_paths:
        target = f"{base_url}{upstream_path}"
        if query:
            target = f"{target}?{query}"
        targets.append(target)
    return tuple(targets)


async def _forward_with_fallback(
    request: Request,
    targets_builder,
) -> Response:
    forward_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length") and k.lower() not in _HOP_BY_HOP
    }
    forward_headers.setdefault("x-forwarded-prefix", "/mlflow")
    forward_headers.setdefault("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("host")
    if host:
        forward_headers.setdefault("x-forwarded-host", host)
    # Ask upstream for identity encoding. Even if upstream still compresses,
    # we sanitize representation headers before returning to clients.
    forward_headers["accept-encoding"] = "identity"

    # Do not forward conditional cache validators for /mlflow requests.
    # We need a full HTML body so asset URL rewriting can run reliably.
    if request.url.path.startswith("/mlflow"):
        for header_name in _CONDITIONAL_CACHE_HEADERS:
            forward_headers.pop(header_name, None)

    body = await request.body()

    response: httpx.Response | None = None
    timeout_failures: list[str] = []

    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        for base_url in MLFLOW_INTERNAL_URLS:
            targets = targets_builder(base_url, request.url.query)
            for index, target in enumerate(targets):
                try:
                    candidate = await client.request(
                        method=request.method,
                        url=target,
                        headers=forward_headers,
                        content=body,
                    )
                except httpx.ConnectError as exc:
                    logger.warning("Cannot connect to MLFlow at %s (%s)", base_url, exc)
                    break
                except httpx.TimeoutException as exc:
                    logger.warning("Timeout proxying to MLFlow at %s (%s)", target, exc)
                    timeout_failures.append(target)
                    break

                if candidate.status_code == 404 and index < len(targets) - 1:
                    logger.info(
                        "MLFlow route %s returned 404, trying fallback route",
                        target,
                    )
                    continue

                response = candidate
                break

            if response is not None:
                break

    if response is None:
        if timeout_failures:
            logger.error("MLFlow timeout via all upstreams: %s", MLFLOW_INTERNAL_URLS)
            return Response(content=b"MLFlow timeout", status_code=504)
        logger.error(
            "Cannot connect to MLFlow via any upstream %s", MLFLOW_INTERNAL_URLS
        )
        return Response(content=b"MLFlow unavailable", status_code=503)

    content = response.content
    content_type = response.headers.get("content-type")
    if request.url.path.startswith("/mlflow"):
        content = _rewrite_mlflow_html(content, content_type)

    resp_headers = {
        k: v for k, v in response.headers.items() if k.lower() not in _HOP_BY_HOP
    }
    for header_name in _UNSAFE_PAYLOAD_HEADERS:
        resp_headers.pop(header_name, None)

    if request.url.path.startswith("/mlflow") and content_type:
        if "text/html" in content_type.lower():
            # Prevent stale HTML cache serving old /static-files paths.
            resp_headers["cache-control"] = "no-store, max-age=0"
            resp_headers.pop("etag", None)
            resp_headers.pop("last-modified", None)

    if content != response.content:
        # Keep cache metadata coherent when HTML body gets rewritten.
        resp_headers.pop("etag", None)

    return Response(
        content=content,
        status_code=response.status_code,
        headers=resp_headers,
        media_type=content_type,
    )


async def _proxy(request: Request, path: str) -> Response:
    """Shared proxy logic: forward request to MLFlow and return its response."""
    await require_admin(request)

    return await _forward_with_fallback(
        request,
        lambda base_url, query: _build_targets(base_url, path, query),
    )


async def _proxy_static(request: Request, path: str) -> Response:
    """Proxy MLFlow static files for legacy absolute /static-files URLs."""
    await require_admin(request)

    return await _forward_with_fallback(
        request,
        lambda base_url, query: _build_static_targets(base_url, path, query),
    )


@router.api_route("", methods=_METHODS)
@router.api_route("/", methods=_METHODS)
async def proxy_mlflow_root(request: Request):
    """Proxy /mlflow and /mlflow/ to the MLFlow root."""
    return await _proxy(request, "")


@router.api_route("/{path:path}", methods=_METHODS)
async def proxy_mlflow(path: str, request: Request):
    """Proxy /mlflow/{path} to the MLFlow tracking server."""
    return await _proxy(request, path)


@static_router.api_route("/static-files", methods=_METHODS)
@static_router.api_route("/static-files/", methods=_METHODS)
async def proxy_mlflow_static_root(request: Request):
    """Proxy /static-files and /static-files/ to MLFlow static bundle roots."""
    return await _proxy_static(request, "")


@static_router.api_route("/static-files/{path:path}", methods=_METHODS)
async def proxy_mlflow_static(path: str, request: Request):
    """Proxy /static-files/{path} to the MLFlow tracking server static files."""
    return await _proxy_static(request, path)
