from typing import Protocol, Any, Dict, List, Optional


class CollectionRepository(Protocol):
    """Protocol for simple document collection repositories (Mongo/Cosmos)."""

    def list_collections(self, database: Optional[str] = None) -> List[str]:
        raise NotImplementedError()

    def query(
        self,
        database: str,
        collection: str,
        filter: Dict[str, Any],
        projection: Optional[Dict[str, int]] = None,
        limit: int = 100,
        skip: int = 0,
        sort: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError()

    def insert(self, database: str, collection: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError()

    def bulk_upsert(
        self,
        database: str,
        collection: str,
        documents: List[Dict[str, Any]],
        ordered: bool = False,
        upsert: bool = True,
    ) -> Dict[str, Any]:
        raise NotImplementedError()


class KeyValueRepository(Protocol):
    """Protocol for key/value stores (Redis)."""

    def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError()

    def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        raise NotImplementedError()

    def delete(self, key: str) -> int:
        raise NotImplementedError()

    def keys(self, pattern: str = "*") -> List[str]:
        raise NotImplementedError()

    def ttl(self, key: str) -> int:
        raise NotImplementedError()


class VectorRepository(Protocol):
    """Protocol for vector stores (Qdrant)."""

    def list_collections(self) -> List[str]:
        raise NotImplementedError()

    def upsert_points(self, collection: str, points: List[Dict[str, Any]]) -> Dict[str, Any]:
        raise NotImplementedError()

    def search(
        self,
        collection: str,
        vector: List[float],
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError()
