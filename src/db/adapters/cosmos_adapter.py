"""Cosmos adapter - thin wrapper around MongoAdapter using Cosmos client."""

from typing import Dict, Any, List, Optional
from src.db.adapters.interfaces import CollectionRepository
from src.db.config import db_clients
from src.db.adapters.mongo_adapter import MongoAdapter


class CosmosAdapter(CollectionRepository):
    """Wraps MongoAdapter but targets the Cosmos DB client and configured DB name."""

    def __init__(self, client=None):
        # Use cosmos client from db.config if none provided
        self.client = client or db_clients.cosmos
        self._inner = MongoAdapter(client=self.client)

    def list_collections(self, database: Optional[str] = None) -> List[str]:
        return self._inner.list_collections(
            database=database or db_clients.config.cosmos_db_name
        )

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
        return self._inner.query(
            database or db_clients.config.cosmos_db_name,
            collection,
            filter,
            projection,
            limit,
            skip,
            sort,
        )

    def insert(
        self, database: str, collection: str, documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        return self._inner.insert(
            database or db_clients.config.cosmos_db_name, collection, documents
        )

    def bulk_upsert(
        self,
        database: str,
        collection: str,
        documents: List[Dict[str, Any]],
        ordered: bool = False,
        upsert: bool = True,
    ) -> Dict[str, Any]:
        return self._inner.bulk_upsert(
            database or db_clients.config.cosmos_db_name,
            collection,
            documents,
            ordered,
            upsert,
        )
