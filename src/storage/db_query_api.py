"""
Database Query API - Provides HTTP endpoints for querying databases.

This service owns the database clients (MongoDB, Redis, Qdrant, Cosmos, etc.)
and exposes concise HTTP endpoints for querying and ingesting data. The app
initializes clients on startup via `startup_db_clients` and tears them down on
shutdown via `shutdown_db_clients`.

Request Flow
------------
- External clients / web frontend:
  1. External clients call the API Gateway at `/api/db/{service}/...` (for
      example, `/api/db/mongodb/collections`).
  2. The Gateway validates the request (API keys, rate limits, sessions) and
      forwards a proxied HTTP request to the internal `db-query-api` service at
      `http://db-query-api:8080/api/{service}/...`.
  3. `db-query-api` receives the proxied request, uses its initialized DB
      clients to perform operations, and returns the result.

- Internal services (workers, batch jobs, other backend services):
  1. Trusted internal services can call `db-query-api` directly at
      `http://db-query-api:8080/api/{service}/...` without going through the
      Gateway. Direct calls are subject to deployment-level network and
      authentication controls (for example, internal-only networks, mTLS, or
      shared secrets).
  2. Direct calls bypass Gateway middleware (API key enforcement) unless the
      deployment puts the Gateway in front of internal traffic as well.

Security and deployment notes
-----------------------------
- For public-facing endpoints, the Gateway centralizes API key validation,
  rate limiting, and session handling. The Gateway should enforce per-service
  API keys for `/api/db/*` endpoints.
- `db-query-api` should never expose database credentials to clients. Keep the
  service reachable only from trusted networks (for example, Kubernetes
  cluster-internal DNS or a private VPC).

"""

from datetime import datetime, timezone
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import JSONResponse
from src.db.clients import (
    startup_db_clients,
    shutdown_db_clients,
    get_mongo_client,
    get_redis_client,
    get_qdrant_client,
    get_cosmos_client,
)
from src.storage.routers import (
    mongodb_router,
    redis_router,
    qdrant_router,
    batch_router,
    cosmos_router,
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Git-Query Database API",
    description="API for querying and ingesting data into databases",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register startup and shutdown events
app.add_event_handler("startup", startup_db_clients)
app.add_event_handler("shutdown", shutdown_db_clients)

# Include routers under /api (database routers expose their own service prefixes)
app.include_router(mongodb_router.router, prefix="/api")
app.include_router(redis_router.router, prefix="/api")
app.include_router(qdrant_router.router, prefix="/api")
app.include_router(batch_router.router, prefix="/api")
app.include_router(cosmos_router.router, prefix="/api")


# ============================================================================
# Health Check
# ============================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint - root level for load balancers"""
    status = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "databases": {
            "mongodb": get_mongo_client() is not None,
            "cosmos": get_cosmos_client() is not None,
            "redis": get_redis_client() is not None,
            "qdrant": get_qdrant_client() is not None,
        },
    }
    return status


@app.get("/api/health")
async def health_check_api():
    """Health check endpoint - API"""
    status = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "databases": {
            "mongodb": get_mongo_client() is not None,
            "redis": get_redis_client() is not None,
            "qdrant": get_qdrant_client() is not None,
        },
    }
    return status


@app.get("/api/health/databases")
async def health_check_databases():
    """Detailed database health check"""
    mongo_client = get_mongo_client()
    cosmos_client = get_cosmos_client()
    redis_client = get_redis_client()
    qdrant_client = get_qdrant_client()

    databases = {
        "mongodb": {
            "connected": mongo_client is not None,
            "status": "healthy" if mongo_client else "unavailable",
        },
        "cosmos": {
            "connected": cosmos_client is not None,
            "status": "healthy" if cosmos_client else "unavailable",
        },
        "redis": {
            "connected": redis_client is not None,
            "status": "healthy" if redis_client else "unavailable",
        },
        "qdrant": {
            "connected": qdrant_client is not None,
            "status": "healthy" if qdrant_client else "unavailable",
        },
    }

    all_healthy = all(db["connected"] for db in databases.values())

    return {
        "status": "healthy" if all_healthy else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "databases": databases,
    }


# ============================================================================
# Documentation Endpoint
# ============================================================================


@app.get("/api/docs/examples")
async def get_examples():
    """Get example queries and usage patterns"""
    return {
        "mongodb_query": {
            "endpoint": "POST /api/mongodb/query",
            "auth": "X-API-Key header required",
            "example": {
                "collection": "users",
                "filter": {"username": "johndoe"},
                "limit": 10,
            },
        },
        "mongodb_insert": {
            "endpoint": "POST /api/mongodb/insert",
            "auth": "X-API-Key header required",
            "example": {
                "collection": "users",
                "documents": [
                    {"username": "alice", "email": "alice@example.com"},
                    {"username": "bob", "email": "bob@example.com"},
                ],
            },
        },
        "qdrant_search": {
            "endpoint": "POST /api/qdrant/search",
            "auth": "X-API-Key header required",
            "example": {
                "collection": "repository_embeddings",
                "vector": [0.1] * 768,
                "limit": 5,
            },
        },
        "qdrant_insert": {
            "endpoint": "POST /api/qdrant/insert",
            "auth": "X-API-Key header required",
            "example": {
                "collection": "repository_embeddings",
                "points": [
                    {
                        "id": 1,
                        "vector": [0.1] * 768,
                        "payload": {"repo_name": "example/repo"},
                    }
                ],
            },
        },
        "batch_insert": {
            "endpoint": "POST /api/batch/insert",
            "auth": "X-API-Key header required",
            "example": {
                "mongodb_data": [
                    {
                        "database": "gitquery",
                        "collection": "repos",
                        "documents": [{"name": "test"}],
                    }
                ],
                "redis_data": [{"key": "cache:test", "value": "data", "ttl": 3600}],
            },
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)


from starlette.exceptions import HTTPException as StarletteHTTPException


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Return JSON for HTTP errors and log route-not-found details."""
    msg = (
        f"HTTP {exc.status_code} on {request.method} {request.url.path} - {exc.detail}"
    )
    if exc.status_code == 404:
        logger.warning(msg)
    else:
        logger.info(msg)

    return JSONResponse(
        status_code=exc.status_code, content={"detail": exc.detail or "Not Found"}
    )


# Simple HTTP exception handler to ensure JSON response for 404s
from starlette.exceptions import HTTPException as StarletteHTTPException


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code, content={"detail": exc.detail or "Not Found"}
    )
