"""Compatibility DB router (legacy alias).

Provides an in-process alias at `/api/db/*` that mounts the storage package
routers (mongodb, redis, qdrant, batch). This preserves the legacy
`/api/db/...` paths while the canonical routes remain under `/api/{service}/...`.
"""

from fastapi import APIRouter, Request

from src.storage.routers import (
    mongodb_router,
    redis_router,
    qdrant_router,
)

# Delegate DB health checks to the canonical health module so both
# `/api/health` and `/api/db/health` report the same information.
from src.gateway.routers.health import health_check_databases


router = APIRouter(prefix="/api/db", tags=["database"])

# Expose the same storage routers under the legacy `/api/db/{service}` paths.
# For example, `/api/db/mongodb/...` will behave the same as `/api/mongodb/...`.
router.include_router(mongodb_router.router, prefix="/mongodb")
router.include_router(redis_router.router, prefix="/redis")
router.include_router(qdrant_router.router, prefix="/qdrant")


@router.get("/health", include_in_schema=True)
async def db_health(request: Request):
    """Check health of all configured database clients.

    Returns a per-service status and overall status. This endpoint is
    intentionally public so orchestration systems can probe service health.
    """
    # Delegate to the canonical implementation in `health.py` which returns a
    # JSONResponse containing per-database details and overall status. This
    # keeps `/api/db/health` and `/api/health` consistent.
    return await health_check_databases(request)
