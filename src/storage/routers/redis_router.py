"""
Redis API endpoints
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Body, Query
from services.db_clients import get_redis_client
from auth import get_api_key

router = APIRouter(prefix="/api/redis", tags=["Redis"])


@router.get("/get/{key}", dependencies=[Depends(get_api_key)])
async def get_redis_key(key: str):
    """Get value from Redis by key (requires API key)"""
    redis_client = get_redis_client()
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")

    try:
        value = redis_client.get(key)
        return {"key": key, "value": value}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Get failed: {str(e)}")


@router.post("/set", dependencies=[Depends(get_api_key)])
async def set_redis_key(
    key: str = Body(...), value: str = Body(...), expire: Optional[int] = Body(None)
):
    """Set value in Redis (requires API key)"""
    redis_client = get_redis_client()
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")

    try:
        if expire:
            redis_client.setex(key, expire, value)
        else:
            redis_client.set(key, value)
        return {"key": key, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Set failed: {str(e)}")


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
