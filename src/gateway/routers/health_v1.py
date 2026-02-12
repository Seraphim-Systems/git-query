"""
Health check endpoints for API v1.

Public endpoints that return the status of all services and databases.
"""

from datetime import datetime, timezone
import httpx
from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/health", tags=["health"])

# Additional route group to expose database health at /api/db/health
db_router = APIRouter(prefix="/api/db", tags=["health"])


@router.get("")
async def health_check_all(request: Request):
    """
    **GET /api/health**

    Returns health status of all services including gateway, databases, and MCP server.

    **Response:**
    ```json
    {
        "status": "healthy",
        "timestamp": "2026-02-08T12:00:00Z",
        "services": {
            "gateway": true,
            "mongodb": true,
            "redis": true,
            "qdrant": true,
            "mcp_server": false
        }
    }
    ```

    **Status Codes:**
    - 200: At least one service is healthy
    - 503: All critical services are down
    """
    try:
        services_status = {
            "gateway": True,  # If we're here, gateway is up
            "mongodb": False,
            "redis": False,
            "qdrant": False,
            "mcp_server": False,
        }

        # Check MongoDB
        try:
            if hasattr(request.app.state, "mongodb"):
                await request.app.state.mongodb.command("ping")
                services_status["mongodb"] = True
        except Exception as e:
            logger.error(f"MongoDB health check failed: {e}")

        # Check Redis
        try:
            if hasattr(request.app.state, "redis"):
                await request.app.state.redis.ping()
                services_status["redis"] = True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")

        # Qdrant health check
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get("http://qdrant:6333/health")
                if resp.status_code == 200:
                    services_status["qdrant"] = True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")

        # Cosmos DB health check (emulator)
        try:
            async with httpx.AsyncClient(timeout=5.0, verify=False) as client:
                resp = await client.get("https://cosmos-db:8081/")
                if resp.status_code in (200, 404, 401):
                    # Emulator may respond with 404 or 401 depending on endpoint
                    services_status["cosmos"] = True
        except Exception as e:
            logger.error(f"Cosmos DB health check failed: {e}")

        # TODO: Add MCP server health check if needed

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


@db_router.get("/health")
async def health_check_databases(request: Request):
    """
    **GET /api/db/health**

    Returns health status of database services only (MongoDB, Redis, Qdrant, Cosmos).
    """
    databases = {}

    # MongoDB
    try:
        if hasattr(request.app.state, "mongodb"):
            await request.app.state.mongodb.command("ping")
            databases["mongodb"] = {"status": True, "url": "mongodb://mongodb:27017"}
    except Exception as e:
        databases["mongodb"] = {"status": False, "error": str(e)}

    # Redis
    try:
        if hasattr(request.app.state, "redis"):
            await request.app.state.redis.ping()
            databases["redis"] = {"status": True, "url": "redis://redis:6379"}
    except Exception as e:
        databases["redis"] = {"status": False, "error": str(e)}

    # Qdrant
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://qdrant:6333/health")
            databases["qdrant"] = {
                "status": resp.status_code == 200,
                "url": "http://qdrant:6333",
                "http_status": resp.status_code,
            }
    except Exception as e:
        databases["qdrant"] = {"status": False, "error": str(e)}

    # Cosmos DB (emulator)
    try:
        async with httpx.AsyncClient(timeout=5.0, verify=False) as client:
            resp = await client.get("https://cosmos-db:8081/")
            databases["cosmos"] = {
                "status": resp.status_code in (200, 404, 401),
                "url": "https://cosmos-db:8081",
                "http_status": resp.status_code,
            }
    except Exception as e:
        databases["cosmos"] = {"status": False, "error": str(e)}

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


# Alias for backward compatibility: expose same logic under the old name.
db_health = health_check_databases
