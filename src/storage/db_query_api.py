"""
Database Query API - Provides HTTP endpoints for querying databases
Designed for data scientists to query and ingest data into the databases
"""

from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from db.clients import (
    startup_db_clients,
    shutdown_db_clients,
    get_mongo_client,
    get_redis_client,
    get_qdrant_client,
    get_cosmos_client,
)
from storage.routers import (
    mongodb_router,
    redis_router,
    qdrant_router,
    batch_router,
    cosmos_router,
)

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

# Include routers with /api/v1 prefix
app.include_router(mongodb_router.router, prefix="/api/v1")
app.include_router(redis_router.router, prefix="/api/v1")
app.include_router(qdrant_router.router, prefix="/api/v1")
app.include_router(batch_router.router, prefix="/api/v1")
app.include_router(cosmos_router.router, prefix="/api/v1")


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


@app.get("/api/v1/health")
async def health_check_v1():
    """Health check endpoint - API v1"""
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


@app.get("/api/v1/health/databases")
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


@app.get("/api/v1/docs/examples")
async def get_examples():
    """Get example queries and usage patterns"""
    return {
        "mongodb_query": {
            "endpoint": "POST /api/v1/mongodb/query",
            "auth": "X-API-Key header required",
            "example": {
                "collection": "users",
                "filter": {"username": "johndoe"},
                "limit": 10,
            },
        },
        "mongodb_insert": {
            "endpoint": "POST /api/v1/mongodb/insert",
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
            "endpoint": "POST /api/v1/qdrant/search",
            "auth": "X-API-Key header required",
            "example": {
                "collection": "repository_embeddings",
                "vector": [0.1] * 768,
                "limit": 5,
            },
        },
        "qdrant_insert": {
            "endpoint": "POST /api/v1/qdrant/insert",
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
            "endpoint": "POST /api/v1/batch/insert",
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
