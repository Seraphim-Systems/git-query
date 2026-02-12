"""
Health check endpoints.

Provides health routes for the API and database services.
"""

from datetime import datetime, timezone
import httpx
import asyncio
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
import logging
from typing import Dict, Any

from src.shared.config import settings
from src.db.clients import get_qdrant_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/health", tags=["health"])


# --- Unified per-service check helpers ---------------------------------
async def _check_mongodb(request: Request) -> Dict[str, Any]:
    try:
        if hasattr(request.app.state, "mongodb"):
            await request.app.state.mongodb.command("ping")
            return {
                "status": True,
                "url": getattr(settings, "mongodb_url", "mongodb://mongodb:27017"),
            }
        return {"status": False, "error": "mongodb client not available"}
    except Exception as e:
        return {"status": False, "error": str(e)}


async def _check_redis(request: Request) -> Dict[str, Any]:
    try:
        if hasattr(request.app.state, "redis"):
            ok = await request.app.state.redis.ping()
            return {
                "status": bool(ok),
                "url": getattr(settings, "redis_url", "redis://redis:6379"),
            }
        return {"status": False, "error": "redis client not available"}
    except Exception as e:
        return {"status": False, "error": str(e)}


async def _check_qdrant(request: Request) -> Dict[str, Any]:
    # Prefer the shared Qdrant client if available, as it performs a lightweight
    # collection check which is more reliable across Qdrant versions.
    qclient = get_qdrant_client()
    url = getattr(settings, "qdrant_url", "http://qdrant:6333")
    if qclient:
        try:
            cols = await asyncio.to_thread(qclient.get_collections)
            count = 0
            try:
                count = len(cols.collections)
            except Exception:
                count = 0
            return {"status": True, "url": url, "collections": count}
        except Exception as e:
            # Fall back to HTTP probe if client call fails
            logger.debug("Qdrant client probe failed, falling back to HTTP: %s", e)

    # Fallback HTTP probe against the collections endpoint (works if /health
    # is not present in this Qdrant build/version).
    collections_url = f"{url.rstrip('/')}/collections"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(collections_url)
            return {
                "status": resp.status_code == 200,
                "url": collections_url,
                "http_status": resp.status_code,
            }
    except Exception as e:
        return {"status": False, "error": str(e), "url": collections_url}


async def _check_cosmos(request: Request) -> Dict[str, Any]:
    url = "https://cosmos-db:8081/"
    # Prefer the initialized Cosmos (Mongo API) client if available. That
    # ensures we probe with the configured credentials instead of making an
    # anonymous HTTP request which will often return 401 for Cosmos instances
    # (especially the emulator). Import at runtime to avoid circular imports.
    try:
        from src.db.clients import get_cosmos_client

        cclient = get_cosmos_client()
        if cclient:
            try:
                # MongoClient is synchronous; run the ping in a thread.
                await asyncio.to_thread(lambda: cclient.admin.command("ping"))
                return {"status": True, "url": getattr(cclient, "address", url)}
            except Exception as e:
                logger.debug("Cosmos Mongo client ping failed: %s", e)
    except Exception:
        # If import or access fails, fall back to HTTP probe below.
        logger.debug("Cosmos client not available for health probe")

    # Fallback HTTP probe (anonymous). Treat 200/404 as healthy; 401 is
    # considered unauthorized (reportable) rather than implicitly healthy.
    try:
        async with httpx.AsyncClient(timeout=5.0, verify=False) as client:
            resp = await client.get(url)
            ok = resp.status_code in (200, 404)
            return {"status": ok, "url": url, "http_status": resp.status_code}
    except Exception as e:
        return {"status": False, "error": str(e), "url": url}


async def _check_mcp_server(request: Request) -> Dict[str, Any]:
    url = getattr(settings, "mcp_server_url", "http://mcp-server:8001")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{url.rstrip('/')}/health")
            return {
                "status": resp.status_code == 200,
                "url": url,
                "http_status": resp.status_code,
            }
    except Exception as e:
        return {"status": False, "error": str(e), "url": url}


# --- Per-service endpoints ---------------------------------------------
@router.get("/mongodb", include_in_schema=True)
async def health_mongodb(request: Request):
    return JSONResponse(
        status_code=(
            status.HTTP_200_OK
            if (await _check_mongodb(request)).get("status")
            else status.HTTP_503_SERVICE_UNAVAILABLE
        ),
        content={"service": "mongodb", "result": await _check_mongodb(request)},
    )


@router.get("/redis", include_in_schema=True)
async def health_redis(request: Request):
    return JSONResponse(
        status_code=(
            status.HTTP_200_OK
            if (await _check_redis(request)).get("status")
            else status.HTTP_503_SERVICE_UNAVAILABLE
        ),
        content={"service": "redis", "result": await _check_redis(request)},
    )


@router.get("/qdrant", include_in_schema=True)
async def health_qdrant(request: Request):
    return JSONResponse(
        status_code=(
            status.HTTP_200_OK
            if (await _check_qdrant(request)).get("status")
            else status.HTTP_503_SERVICE_UNAVAILABLE
        ),
        content={"service": "qdrant", "result": await _check_qdrant(request)},
    )


@router.get("/cosmos", include_in_schema=True)
async def health_cosmos(request: Request):
    return JSONResponse(
        status_code=(
            status.HTTP_200_OK
            if (await _check_cosmos(request)).get("status")
            else status.HTTP_503_SERVICE_UNAVAILABLE
        ),
        content={"service": "cosmos", "result": await _check_cosmos(request)},
    )


@router.get("/mcp", include_in_schema=True)
async def health_mcp(request: Request):
    return JSONResponse(
        status_code=(
            status.HTTP_200_OK
            if (await _check_mcp_server(request)).get("status")
            else status.HTTP_503_SERVICE_UNAVAILABLE
        ),
        content={"service": "mcp", "result": await _check_mcp_server(request)},
    )


@router.get("")
async def health_check_all(request: Request):
    try:
        # Reuse the per-service helpers so checks are consistent.
        mongo_res = await _check_mongodb(request)
        redis_res = await _check_redis(request)
        qdrant_res = await _check_qdrant(request)
        cosmos_res = await _check_cosmos(request)
        mcp_res = await _check_mcp_server(request)

        services_status = {
            "gateway": True,
            "mongodb": bool(mongo_res.get("status")),
            "redis": bool(redis_res.get("status")),
            "qdrant": bool(qdrant_res.get("status")),
            "cosmos": bool(cosmos_res.get("status")),
            "mcp_server": bool(mcp_res.get("status")),
        }

        overall_status = "healthy" if any(services_status.values()) else "unhealthy"
        http_status = (
            status.HTTP_200_OK
            if overall_status == "healthy"
            else status.HTTP_503_SERVICE_UNAVAILABLE
        )

        return JSONResponse(
            status_code=http_status,
            content={
                "status": overall_status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "services": services_status,
                "details": {
                    "mongodb": mongo_res,
                    "redis": redis_res,
                    "qdrant": qdrant_res,
                    "cosmos": cosmos_res,
                    "mcp_server": mcp_res,
                },
            },
        )

    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )


async def health_check_databases(request: Request):
    # Use the same helpers as the public health check so database checks are
    # identical and reliable.
    mongo_res = await _check_mongodb(request)
    redis_res = await _check_redis(request)
    qdrant_res = await _check_qdrant(request)
    cosmos_res = await _check_cosmos(request)

    databases = {
        "mongodb": mongo_res,
        "redis": redis_res,
        "qdrant": qdrant_res,
        "cosmos": cosmos_res,
    }

    overall_healthy = any(db.get("status", False) for db in databases.values())

    return JSONResponse(
        status_code=(
            status.HTTP_200_OK
            if overall_healthy
            else status.HTTP_503_SERVICE_UNAVAILABLE
        ),
        content={
            "status": "healthy" if overall_healthy else "unhealthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "databases": databases,
        },
    )


# Note: `health_check_databases` is intentionally a plain function so the
# legacy `/api/db/health` router can delegate to it (see
# `src/gateway/routers/db.py`). The old `db_router` was unused and has been
# removed to avoid duplicate/unused route definitions.
