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

router = APIRouter(prefix="/api/db", tags=["database"])

# Expose the same storage routers under the legacy `/api/db/{service}` paths.
# For example, `/api/db/mongodb/...` will behave the same as `/api/mongodb/...`.
router.include_router(mongodb_router.router, prefix="/mongodb")
router.include_router(redis_router.router, prefix="/redis")
router.include_router(qdrant_router.router, prefix="/qdrant")
router.include_router(cosmos_router.router, prefix="/cosmos")
