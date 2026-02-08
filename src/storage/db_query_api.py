"""
Database Query API - Provides HTTP endpoints for querying databases
Designed for data scientists to query and ingest data into the databases
"""

from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from services.db_clients import (
    startup_db_clients,
    shutdown_db_clients,
    get_mongo_client,
    get_redis_client,
    get_qdrant_client,
)
from routers import mongodb_router, redis_router, qdrant_router, batch_router

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

# Include routers
app.include_router(mongodb_router.router)
app.include_router(redis_router.router)
app.include_router(qdrant_router.router)
app.include_router(batch_router.router)


# ============================================================================
# Health Check
# ============================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint"""
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
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
