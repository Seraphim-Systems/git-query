"""Mongo / Cosmos adapter implementing CollectionRepository."""

from typing import Dict, Any, List, Optional
from src.db.adapters.interfaces import CollectionRepository
from src.db.config import db_clients


class MongoAdapter(CollectionRepository):
    def __init__(self, client=None):
        self.client = client or db_clients.mongodb

    def list_collections(self, database: Optional[str] = None) -> List[str]:
        dbname = database or db_clients.config.mongodb_db
        db = self.client[dbname]
        return db.list_collection_names()

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
        db = self.client[database]
        coll = db[collection]
        cursor = coll.find(filter, projection).limit(limit).skip(skip)
        if sort:
            cursor = cursor.sort(list(sort.items()))
        docs = list(cursor)
        for d in docs:
            if "_id" in d:
                d["_id"] = str(d["_id"])
        return {"count": len(docs), "documents": docs}

    def insert(
        self, database: str, collection: str, documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        db = self.client[database]
        coll = db[collection]
        result = coll.insert_many(documents)
        return {
            "inserted_count": len(result.inserted_ids),
            "inserted_ids": [str(i) for i in result.inserted_ids],
        }

    def bulk_upsert(
        self,
        database: str,
        collection: str,
        documents: List[Dict[str, Any]],
        ordered: bool = False,
        upsert: bool = True,
    ) -> Dict[str, Any]:
        db = self.client[database]
        coll = db[collection]
        from pymongo import ReplaceOne

        operations = []
        for doc in documents:
            if "_id" in doc:
                operations.append(ReplaceOne({"_id": doc["_id"]}, doc, upsert=upsert))
            else:
                operations.append(ReplaceOne(doc, doc, upsert=True))
        result = coll.bulk_write(operations, ordered=ordered)
        return {
            "matched": getattr(result, "matched_count", None),
            "modified": getattr(result, "modified_count", None),
            "upserted": getattr(result, "upserted_count", None),
            "inserted": len(documents),
        }
