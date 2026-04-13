from typing import Any, Optional, List

from src.db.clients import get_redis_client
from src.db.adapters.interfaces import KeyValueRepository


class RedisAdapter(KeyValueRepository):
    """Simple Redis adapter implementing `KeyValueRepository`.

    The adapter accepts an optional client; if none provided it will use
    the runtime client from `db.clients.get_redis_client()`.
    """

    def __init__(self, client=None):
        self._client = client or get_redis_client()

    def get(self, key: str) -> Optional[Any]:
        if not self._client:
            return None
        return self._client.get(key)

    def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        if not self._client:
            return False
        if expire:
            return self._client.setex(key, expire, value)
        return self._client.set(key, value)

    def delete(self, key: str) -> int:
        if not self._client:
            return 0
        return self._client.delete(key)

    def keys(self, pattern: str = "*") -> List[str]:
        if not self._client:
            return []
        return list(self._client.scan_iter(match=pattern))

    def ttl(self, key: str) -> int:
        if not self._client:
            return -2
        return self._client.ttl(key)
