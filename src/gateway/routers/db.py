"""Compatibility DB router (legacy alias).

Provides an in-process alias at `/api/db/*` that mounts the storage package
routers (mongodb, redis, qdrant, cosmos, batch). This preserves the legacy
`/api/db/...` paths while the canonical routes remain under `/api/{service}/...`.
"""

from fastapi import APIRouter

from src.storage.routers import (
    mongodb_router,
    redis_router,
    qdrant_router,
    cosmos_router,
)
from src.db.clients import (
    get_mongo_client,
    get_redis_client,
    get_qdrant_client,
    get_cosmos_client,
)

router = APIRouter(prefix="/api/db", tags=["database"])

# Expose the same storage routers under the legacy `/api/db/{service}` paths.
# For example, `/api/db/mongodb/...` will behave the same as `/api/mongodb/...`.
router.include_router(mongodb_router.router, prefix="/mongodb")
router.include_router(redis_router.router, prefix="/redis")
router.include_router(qdrant_router.router, prefix="/qdrant")
router.include_router(cosmos_router.router, prefix="/cosmos")


@router.get("/health", include_in_schema=True)
async def db_health():
    """Check health of all configured database clients.

    Returns a per-service status and overall status. This endpoint is
    intentionally public so orchestration systems can probe service health.
    """
    statuses = {}
    overall_ok = True

    # MongoDB
    mongo = get_mongo_client()
    try:
        if mongo:
            mongo.admin.command("ping")
            statuses["mongodb"] = {"status": "ok"}
        else:
            statuses["mongodb"] = {"status": "unavailable"}
            overall_ok = False
    except Exception as e:
        statuses["mongodb"] = {"status": "error", "detail": str(e)}
        overall_ok = False

    # Cosmos (Mongo-compatible client)
    cosmos = get_cosmos_client()
    try:
        if cosmos:
            cosmos.admin.command("ping")
            statuses["cosmos"] = {"status": "ok"}
        else:
            statuses["cosmos"] = {"status": "unavailable"}
            overall_ok = False
    except Exception as e:
        statuses["cosmos"] = {"status": "error", "detail": str(e)}
        overall_ok = False

    # Redis
    redis_client = get_redis_client()
    try:
        if redis_client:
            # redis-py raises on failure; ping returns True on success
            ok = redis_client.ping()
            statuses["redis"] = {"status": "ok"} if ok else {"status": "error"}
            if not ok:
                overall_ok = False
        else:
            statuses["redis"] = {"status": "unavailable"}
            overall_ok = False
    except Exception as e:
        statuses["redis"] = {"status": "error", "detail": str(e)}
        overall_ok = False

    # Qdrant
    qdrant = get_qdrant_client()
    try:
        if qdrant:
            # try a lightweight call
            _ = qdrant.get_collections()
            statuses["qdrant"] = {"status": "ok"}
        else:
            statuses["qdrant"] = {"status": "unavailable"}
            overall_ok = False
    except Exception as e:
        statuses["qdrant"] = {"status": "error", "detail": str(e)}
        overall_ok = False

    return {"status": ("ok" if overall_ok else "degraded"), "services": statuses}
