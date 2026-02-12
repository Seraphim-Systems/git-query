"""
Database proxy router for the gateway.

Provides a unified interface to interact with MongoDB, Redis, Qdrant, and MCP server.
All endpoints require API key authentication (per-service) when accessed through the
API Gateway.

Request Flow
------------
- External clients / web frontend:
  1. Client (browser / mobile / external service) calls the API Gateway at
      `/api/db/{service}/...` (e.g. `/api/db/mongodb/collections`).
  2. `APIKeyMiddleware` runs in the Gateway: public paths are allowed; DB paths
      require a per-service API key. If the key is valid the request proceeds.
  3. The Gateway handles database operations. Historically these were
      implemented by a separate `db-query-api` service; the Gateway now mounts
      the database routers directly under `/api/{service}/...` and serves them
      in-process. The proxy endpoints below forward requests to the internal
      `/api/*` paths.

- Internal services (workers, backends, trusted services):
  1. Trusted internal services running in the same network can call the
      `db-query-api` service directly at `http://db-query-api:8080/api/{service}/...`.
  2. Calls made directly to `db-query-api` typically bypass the Gateway. In
      that case, authentication and authorization expectations depend on
      deployment configuration; for example, internal network policies, mTLS,
      or additional shared secrets may be used. The Gateway's per-service API
      key checks do not apply to intra-cluster direct calls unless the
      infrastructure enforces them.

Notes
-----
- Keep database credentials and drivers inside `db-query-api` to minimize the
  attack surface.
- The Gateway centralizes API key validation and rate-limiting for external
  traffic, while the `db-query-api` focuses on database client management and
  request handling.
"""

from fastapi import APIRouter, HTTPException, Request, Body
from fastapi.responses import JSONResponse
from typing import Any, Dict, List, Optional
import httpx
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/db", tags=["database"])

# Internal service base URL - point to the local mounted `/api` paths.
# When running inside Docker, ensure the Gateway process listens on port 8000
# and container networking allows loopback connections if necessary.
DB_QUERY_API_URL = "http://127.0.0.1:8000/api"


def _build_forward_headers(request: Request) -> dict:
    """Build a minimal set of headers to forward to db-query-api.

    We forward the incoming Authorization or X-API-Key header if present.
    Do not forward any other client headers to avoid leaking internal info.
    """
    headers = {}
    auth = request.headers.get("authorization")
    xkey = request.headers.get("x-api-key") or request.headers.get("X-API-Key")
    if auth:
        headers["Authorization"] = auth
    elif xkey:
        headers["X-API-Key"] = xkey

    # Log presence (masked) for debugging
    if headers:
        for k, v in headers.items():
            logger.debug("Forwarding header %s (len=%d)", k, len(v))
    else:
        logger.debug("No auth header to forward to internal DB handlers")

    return headers


# ============================================================================
# MongoDB Endpoints
# ============================================================================


@router.get("/mongodb/collections")
async def list_mongodb_collections(request: Request):
    """
    **GET /api/db/mongodb/collections**

    List all MongoDB collections.

    **Auth:** Requires MongoDB API key

    **Response:**
    ```json
    {
        "collections": ["repositories", "commits", "users"]
    }
    ```
    """
    headers = _build_forward_headers(request)
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{DB_QUERY_API_URL}/mongodb/collections", headers=headers
        )
        response.raise_for_status()
        return response.json()


@router.get("/cosmos/collections")
async def list_cosmos_collections(request: Request):
    """
    **GET /api/db/cosmos/collections**

    List all Cosmos DB collections.
    """
    headers = _build_forward_headers(request)
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{DB_QUERY_API_URL}/cosmos/collections", headers=headers
        )
        response.raise_for_status()
        return response.json()


@router.post("/mongodb/{collection}/query")
async def query_mongodb_collection(
    collection: str,
    request: Request,
    query: Dict[str, Any] = Body(
        ...,
        example={
            "filter": {"stars": {"$gt": 100}},
            "projection": {"name": 1, "stars": 1},
            "limit": 10,
            "skip": 0,
        },
    ),
):
    """
    **POST /api/db/mongodb/{collection}/query**

    Query MongoDB collection with filters, projection, and pagination.

    **Auth:** Requires MongoDB API key

    **Request Body:**
    ```json
    {
        "filter": {"field": "value"},
        "projection": {"field": 1},
        "limit": 10,
        "skip": 0,
        "sort": {"field": -1}
    }
    ```

    **Response:**
    ```json
    {
        "documents": [...],
        "count": 10
    }
    ```
    """
    headers = _build_forward_headers(request)
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{DB_QUERY_API_URL}/mongodb/{collection}/query",
            json=query,
            headers=headers,
        )
        response.raise_for_status()
        return response.json()


@router.post("/cosmos/{collection}/query")
async def query_cosmos_collection(
    collection: str,
    request: Request,
    query: Dict[str, Any] = Body(
        ...,
        example={
            "filter": {"stars": {"$gt": 100}},
            "projection": {"name": 1, "stars": 1},
            "limit": 10,
            "skip": 0,
        },
    ),
):
    headers = _build_forward_headers(request)
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{DB_QUERY_API_URL}/cosmos/{collection}/query", json=query, headers=headers
        )
        response.raise_for_status()
        return response.json()


@router.post("/mongodb/{collection}/bulk")
async def bulk_insert_mongodb(
    collection: str,
    request: Request,
    payload: Dict[str, Any] = Body(
        ...,
        example={
            "documents": [
                {"_id": "1", "name": "repo1", "stars": 100},
                {"_id": "2", "name": "repo2", "stars": 200},
            ],
            "ordered": False,
            "upsert": True,
        },
    ),
):
    """
    **POST /api/db/mongodb/{collection}/bulk**

    Bulk insert or upsert documents into MongoDB collection.
    Optimized for loading large datasets.

    **Auth:** Requires MongoDB API key

    **Request Body:**
    ```json
    {
        "documents": [{"_id": "1", "data": "..."}],
        "ordered": false,
        "upsert": true
    }
    ```

    **Response:**
    ```json
    {
        "inserted": 150,
        "updated": 50,
        "errors": []
    }
    ```
    """
    headers = _build_forward_headers(request)
    async with httpx.AsyncClient(
        timeout=300.0
    ) as client:  # 5 min timeout for large batches
        response = await client.post(
            f"{DB_QUERY_API_URL}/mongodb/{collection}/bulk",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        return response.json()


@router.post("/cosmos/{collection}/bulk")
async def bulk_insert_cosmos(
    collection: str,
    request: Request,
    payload: Dict[str, Any] = Body(
        ...,
        example={
            "documents": [
                {"_id": "1", "name": "repo1", "stars": 100},
                {"_id": "2", "name": "repo2", "stars": 200},
            ],
            "ordered": False,
            "upsert": True,
        },
    ),
):
    headers = _build_forward_headers(request)
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            f"{DB_QUERY_API_URL}/cosmos/{collection}/bulk",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        return response.json()


# ============================================================================
# Redis Endpoints
# ============================================================================


@router.get("/redis/{key}")
async def get_redis_key(key: str, request: Request):
    """
    **GET /api/db/redis/{key}**

    Get value for a Redis key.

    **Auth:** Requires Redis API key

    **Response:**
    ```json
    {
        "key": "cache:user:123",
        "value": "...",
        "ttl": 3600
    }
    ```
    """
    headers = _build_forward_headers(request)
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{DB_QUERY_API_URL}/redis/{key}", headers=headers)
        response.raise_for_status()
        return response.json()


@router.post("/redis/batch")
async def redis_batch_operations(
    request: Request,
    operations: Dict[str, Any] = Body(
        ...,
        example={
            "operations": [
                {"action": "set", "key": "cache:1", "value": "data1", "ttl": 3600},
                {"action": "get", "key": "cache:2"},
            ]
        },
    ),
):
    """
    **POST /api/db/redis/batch**

    Execute batch Redis operations (get, set, delete).

    **Auth:** Requires Redis API key

    **Request Body:**
    ```json
    {
        "operations": [
            {"action": "set", "key": "k1", "value": "v1", "ttl": 3600},
            {"action": "get", "key": "k2"},
            {"action": "delete", "key": "k3"}
        ]
    }
    ```

    **Response:**
    ```json
    {
        "results": [
            {"key": "k1", "status": "ok"},
            {"key": "k2", "value": "v2"},
            {"key": "k3", "status": "deleted"}
        ]
    }
    ```
    """
    headers = _build_forward_headers(request)
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            f"{DB_QUERY_API_URL}/redis/batch", json=operations, headers=headers
        )
        response.raise_for_status()
        return response.json()


# ============================================================================
# Qdrant Endpoints
# ============================================================================


@router.get("/qdrant/collections")
async def list_qdrant_collections(request: Request):
    """
    **GET /api/db/qdrant/collections**

    List all Qdrant collections.

    **Auth:** Requires Qdrant API key

    **Response:**
    ```json
    {
        "collections": ["embeddings", "vectors"]
    }
    ```
    """
    headers = _build_forward_headers(request)
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{DB_QUERY_API_URL}/qdrant/collections", headers=headers
        )
        response.raise_for_status()
        return response.json()


@router.post("/qdrant/{collection}/search")
async def search_qdrant_vectors(
    collection: str,
    request: Request,
    query: Dict[str, Any] = Body(
        ...,
        example={
            "vector": [0.1, 0.2, 0.3],
            "limit": 10,
            "filter": {},
            "with_payload": True,
        },
    ),
):
    """
    **POST /api/db/qdrant/{collection}/search**

    Search for similar vectors in Qdrant collection.

    **Auth:** Requires Qdrant API key

    **Request Body:**
    ```json
    {
        "vector": [0.1, 0.2, ...],
        "limit": 10,
        "filter": {},
        "with_payload": true
    }
    ```

    **Response:**
    ```json
    {
        "results": [
            {"id": "vec1", "score": 0.95, "payload": {...}}
        ]
    }
    ```
    """
    headers = _build_forward_headers(request)
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{DB_QUERY_API_URL}/qdrant/{collection}/search",
            json=query,
            headers=headers,
        )
        response.raise_for_status()
        return response.json()


@router.post("/qdrant/{collection}/bulk")
async def bulk_upsert_qdrant(
    collection: str,
    request: Request,
    payload: Dict[str, Any] = Body(
        ...,
        example={
            "points": [
                {"id": "1", "vector": [0.1, 0.2], "payload": {"name": "item1"}},
                {"id": "2", "vector": [0.3, 0.4], "payload": {"name": "item2"}},
            ],
            "wait": True,
        },
    ),
):
    """
    **POST /api/db/qdrant/{collection}/bulk**

    Bulk upsert vectors into Qdrant collection.
    Optimized for loading large vector datasets.

    **Auth:** Requires Qdrant API key

    **Request Body:**
    ```json
    {
        "points": [
            {"id": "1", "vector": [...], "payload": {...}}
        ],
        "wait": true
    }
    ```

    **Response:**
    ```json
    {
        "upserted": 100,
        "status": "completed"
    }
    ```
    """
    headers = _build_forward_headers(request)
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            f"{DB_QUERY_API_URL}/qdrant/{collection}/bulk",
            json=payload,
            headers=headers,
        )
        response.raise_for_status()
        return response.json()


# ============================================================================
# MCP Server Endpoints
# ============================================================================


@router.get("/mcp/tools")
async def list_mcp_tools(request: Request):
    """
    **GET /api/db/mcp/tools**

    List available MCP tools (including recommendation engine).

    **Auth:** Requires MCP API key

    **Response:**
    ```json
    {
        "tools": [
            {"name": "recommend", "description": "Get recommendations"},
            {"name": "example_tool", "description": "Example tool"}
        ]
    }
    ```
    """
    headers = _build_forward_headers(request)
    # Forward to internal MCP handlers (served by the Gateway or local services)
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{DB_QUERY_API_URL}/mcp/tools", headers=headers
            )
            response.raise_for_status()
            return response.json()
    except Exception:
        return {
            "tools": [],
            "note": "MCP integration pending or internal MCP service unreachable",
        }


@router.post("/mcp/tools/{tool_name}")
async def execute_mcp_tool(
    tool_name: str, request: Request, params: Dict[str, Any] = Body(...)
):
    """
    **POST /api/db/mcp/tools/{tool_name}**

    Execute an MCP tool (e.g., recommendation engine).

    **Auth:** Requires MCP API key

    **Request Body:**
    ```json
    {
        "user_id": "123",
        "context": {...}
    }
    ```
    """
    headers = _build_forward_headers(request)
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{DB_QUERY_API_URL}/mcp/tools/{tool_name}",
                headers=headers,
                json=params,
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
