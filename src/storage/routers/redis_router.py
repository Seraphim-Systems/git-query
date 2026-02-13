"""
Redis API endpoints
"""

from typing import Dict, Any
import json
from fastapi import APIRouter, HTTPException, Depends, Body, Query
from src.db.clients import get_redis_client
from src.storage.auth import get_api_key

router = APIRouter(prefix="/redis", tags=["Redis"])


@router.get("/{key}", dependencies=[Depends(get_api_key)])
async def get_redis_key(key: str):
    """Get value from Redis by key (requires API key)"""
    redis_client = get_redis_client()
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")

    try:
        raw = redis_client.get(key)
        ttl = redis_client.ttl(key)

        # Normalize binary values to string for JSON responses
        if isinstance(raw, (bytes, bytearray)):
            try:
                value = raw.decode("utf-8")
            except Exception:
                value = raw.decode("utf-8", errors="replace")
        else:
            value = raw

        return {"key": key, "value": value, "ttl": ttl if ttl and ttl > 0 else None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Get failed: {str(e)}")


@router.put("/{key}", dependencies=[Depends(get_api_key)])
async def set_redis_key(
    key: str,
    payload: Any = Body(..., example={"value": "data", "ttl": 3600}),
):
    """Set value in Redis with optional TTL (requires API key)"""
    redis_client = get_redis_client()
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")

    try:
        # Accept either a JSON object {value, ttl} or a raw string body
        ttl = None
        if isinstance(payload, dict):
            value = payload.get("value")
            ttl = payload.get("ttl")
        else:
            # Could be a raw string or number from CLI; convert to string
            value = payload

        # If value is a complex object, serialize as JSON string
        if value is not None and not isinstance(value, (str, bytes, bytearray, int, float, bool)):
            value = json.dumps(value)

        if ttl:
            redis_client.setex(key, int(ttl), value)
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
                raw = redis_client.get(key)
                if isinstance(raw, (bytes, bytearray)):
                    try:
                        val = raw.decode("utf-8")
                    except Exception:
                        val = raw.decode("utf-8", errors="replace")
                else:
                    val = raw
                results.append({"key": key, "value": val, "status": "ok"})

            elif action == "set":
                value = op.get("value")
                ttl = op.get("ttl")
                if value is not None and not isinstance(value, (str, bytes, bytearray, int, float, bool)):
                    value = json.dumps(value)
                if ttl:
                    redis_client.setex(key, int(ttl), value)
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
        for raw_key in redis_client.scan_iter(match=pattern, count=limit):
            if isinstance(raw_key, (bytes, bytearray)):
                try:
                    k = raw_key.decode("utf-8")
                except Exception:
                    k = raw_key.decode("utf-8", errors="replace")
            else:
                k = raw_key
            keys.append(k)
            if len(keys) >= limit:
                break
        return {"pattern": pattern, "count": len(keys), "keys": keys}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Keys listing failed: {str(e)}")
