from typing import Protocol, Any, Dict, List, Optional


class CollectionRepository(Protocol):
    """Protocol for simple document collection repositories (Mongo/Cosmos)."""

    def list_collections(self, database: Optional[str] = None) -> List[str]: ...

    def query(
        self,
        database: str,
        collection: str,
        filter: Dict[str, Any],
        projection: Optional[Dict[str, int]] = None,
        limit: int = 100,
        skip: int = 0,
        sort: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]: ...

    def insert(
        self, database: str, collection: str, documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]: ...

    def bulk_upsert(
        self,
        database: str,
        collection: str,
        documents: List[Dict[str, Any]],
        ordered: bool = False,
        upsert: bool = True,
    ) -> Dict[str, Any]: ...


class KeyValueRepository(Protocol):
    """Protocol for key/value stores (Redis)."""

    def get(self, key: str) -> Optional[Any]: ...

    def set(self, key: str, value: Any, expire: Optional[int] = None) -> bool: ...

    def delete(self, key: str) -> int: ...

    def keys(self, pattern: str = "*") -> List[str]: ...

    def ttl(self, key: str) -> int: ...


class VectorRepository(Protocol):
    """Protocol for vector stores (Qdrant)."""

    def list_collections(self) -> List[str]: ...

    def upsert_points(self, collection: str, points: List[Dict[str, Any]]) -> Dict[str, Any]: ...

    def search(
        self,
        collection: str,
        vector: List[float],
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]: ...
