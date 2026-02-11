"""
Redis API endpoints
"""

from typing import Optional, Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends, Body, Query
from db.clients import get_redis_client
from auth import get_api_key

router = APIRouter(prefix="/redis", tags=["Redis"])


@router.get("/{key}", dependencies=[Depends(get_api_key)])
async def get_redis_key(key: str):
    """Get value from Redis by key (requires API key)"""
    redis_client = get_redis_client()
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")

    try:
        value = redis_client.get(key)
        ttl = redis_client.ttl(key)
        return {"key": key, "value": value, "ttl": ttl if ttl > 0 else None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Get failed: {str(e)}")


@router.put("/{key}", dependencies=[Depends(get_api_key)])
async def set_redis_key(
    key: str,
    payload: Dict[str, Any] = Body(..., example={"value": "data", "ttl": 3600}),
):
    """Set value in Redis with optional TTL (requires API key)"""
    redis_client = get_redis_client()
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")

    try:
        value = payload.get("value")
        ttl = payload.get("ttl")

        if ttl:
            redis_client.setex(key, ttl, value)
        else:
            redis_client.set(key, value)
        return {"key": key, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Set failed: {str(e)}")


@router.post("/batch", dependencies=[Depends(get_api_key)])
async def redis_batch_operations(
    operations: Dict[str, Any] = Body(
        ...,
        example={
            "operations": [
                {"action": "set", "key": "k1", "value": "v1", "ttl": 3600},
                {"action": "get", "key": "k2"},
                {"action": "delete", "key": "k3"},
            ]
        },
    )
):
    """
    Execute batch Redis operations.

    Supports actions: get, set, delete

    Args:
        operations: {
            "operations": [
                {"action": "set", "key": "k1", "value": "v1", "ttl": 3600},
                {"action": "get", "key": "k2"},
                {"action": "delete", "key": "k3"}
            ]
        }
    """
    redis_client = get_redis_client()
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")

    try:
        results = []
        ops = operations.get("operations", [])

        for op in ops:
            action = op.get("action")
            key = op.get("key")

            if action == "get":
                value = redis_client.get(key)
                results.append({"key": key, "value": value, "status": "ok"})

            elif action == "set":
                value = op.get("value")
                ttl = op.get("ttl")
                if ttl:
                    redis_client.setex(key, ttl, value)
                else:
                    redis_client.set(key, value)
                results.append({"key": key, "status": "ok"})

            elif action == "delete":
                redis_client.delete(key)
                results.append({"key": key, "status": "deleted"})

            else:
                results.append(
                    {
                        "key": key,
                        "status": "error",
                        "error": f"Unknown action: {action}",
                    }
                )

        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch operation failed: {str(e)}")


@router.get("/keys", dependencies=[Depends(get_api_key)])
async def list_redis_keys(pattern: str = "*", limit: int = Query(default=100, le=1000)):
    """
    List Redis keys matching pattern (requires API key).

    WARNING: This endpoint uses SCAN which can be expensive for large keyspaces.
    Always use specific patterns (e.g., 'user:*') instead of '*' when possible.
    The limit parameter controls how many keys are returned, but the scan may
    examine many more keys internally. For production use with millions of keys,
    consider using more specific patterns or querying smaller key namespaces.

    Args:
        pattern: Redis key pattern (default: "*")
        limit: Maximum number of keys to return (default: 100, max: 1000)
    """
    redis_client = get_redis_client()
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")

    try:
        keys = []
        for key in redis_client.scan_iter(match=pattern, count=limit):
            keys.append(key)
            if len(keys) >= limit:
                break
        return {"pattern": pattern, "count": len(keys), "keys": keys}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Keys listing failed: {str(e)}")
