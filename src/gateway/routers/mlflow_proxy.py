"""MLFlow reverse-proxy router.

Forwards all /mlflow/* requests to the MLFlow container on the internal
Docker network.  Admin session authentication is required.
"""

import logging

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import Response

from src.gateway.middleware.session import require_admin

logger = logging.getLogger(__name__)

MLFLOW_INTERNAL_URL = "http://mlflow:5000"

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

_METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]

router = APIRouter(tags=["MLFlow"])


async def _proxy(request: Request, path: str) -> Response:
    """Shared proxy logic: forward request to MLFlow and return its response."""
    await require_admin(request)

    target = f"{MLFLOW_INTERNAL_URL}/{path}"
    if request.url.query:
        target = f"{target}?{request.url.query}"

    forward_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length") and k.lower() not in _HOP_BY_HOP
    }

    body = await request.body()

    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        try:
            resp = await client.request(
                method=request.method,
                url=target,
                headers=forward_headers,
                content=body,
            )
        except httpx.ConnectError:
            logger.error("Cannot connect to MLFlow at %s", MLFLOW_INTERNAL_URL)
            return Response(content=b"MLFlow unavailable", status_code=503)
        except httpx.TimeoutException:
            logger.error("Timeout proxying to MLFlow at %s", target)
            return Response(content=b"MLFlow timeout", status_code=504)

    resp_headers = {k: v for k, v in resp.headers.items() if k.lower() not in _HOP_BY_HOP}

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=resp_headers,
        media_type=resp.headers.get("content-type"),
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
