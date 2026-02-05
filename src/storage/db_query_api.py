"""
Database Query API - Provides HTTP endpoints for querying databases
Designed for data scientists to query and ingest data into the databases
"""

import os
import json
from typing import Any, Dict, List, Optional
from datetime import datetime
from fastapi import FastAPI, HTTPException, Depends, Security, Body, Query
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
import pymongo
from pymongo import MongoClient
import redis
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams

# Initialize FastAPI app
app = FastAPI(
    title="Git-Query Database API",
    description="API for querying and ingesting data into databases",
    version="1.0.0",
)

# Configuration
MONGODB_URL = os.getenv(
    "MONGODB_URL", "mongodb://admin:mongopass@localhost:27017/gitquery?authSource=admin"
)
REDIS_URL = os.getenv("REDIS_URL", "redis://:redispass@localhost:6379")
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
DATA_INGESTION_API_KEY = os.getenv("DATA_INGESTION_API_KEY", "change-me")

# Security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Database clients
mongo_client: Optional[MongoClient] = None
redis_client: Optional[redis.Redis] = None
qdrant_client: Optional[QdrantClient] = None


def get_api_key(api_key: Optional[str] = Security(api_key_header)) -> str:
    """Validate API key for data ingestion endpoints"""
    if api_key == DATA_INGESTION_API_KEY:
        return api_key
    raise HTTPException(status_code=403, detail="Invalid or missing API key")


@app.on_event("startup")
async def startup_db_clients():
    """Initialize database clients on startup"""
    global mongo_client, redis_client, qdrant_client

    try:
        client = MongoClient(MONGODB_URL)
        # Test connection
        client.admin.command("ping")
        mongo_client = client
        print("✓ MongoDB connected")
    except Exception as e:
        mongo_client = None
        print(f"✗ MongoDB connection failed: {e}")

    try:
        client = redis.from_url(REDIS_URL, decode_responses=True)
        client.ping()
        redis_client = client
        print("✓ Redis connected")
    except Exception as e:
        redis_client = None
        print(f"✗ Redis connection failed: {e}")

    try:
        client = QdrantClient(
            host=QDRANT_HOST, port=QDRANT_PORT, api_key=QDRANT_API_KEY
        )
        client.get_collections()
        qdrant_client = client
        print("✓ Qdrant connected")
    except Exception as e:
        qdrant_client = None
        print(f"✗ Qdrant connection failed: {e}")


@app.on_event("shutdown")
async def shutdown_db_clients():
    """Close database connections on shutdown"""
    global mongo_client, redis_client, qdrant_client

    if mongo_client:
        mongo_client.close()
    if redis_client:
        redis_client.close()
    if qdrant_client:
        qdrant_client.close()


# ============================================================================
# Health Check
# ============================================================================


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "databases": {
            "mongodb": mongo_client is not None,
            "redis": redis_client is not None,
            "qdrant": qdrant_client is not None,
        },
    }
    return status


# ============================================================================
# MongoDB Query Endpoints
# ============================================================================


class MongoQuery(BaseModel):
    database: str = Field(default="gitquery", description="Database name")
    collection: str = Field(..., description="Collection name")
    filter: Dict[str, Any] = Field(default={}, description="Query filter")
    projection: Optional[Dict[str, int]] = Field(
        default=None, description="Fields to return"
    )
    limit: int = Field(default=100, le=1000, description="Maximum number of documents")
    skip: int = Field(default=0, description="Number of documents to skip")
    sort: Optional[Dict[str, int]] = Field(default=None, description="Sort criteria")


class MongoInsert(BaseModel):
    database: str = Field(default="gitquery", description="Database name")
    collection: str = Field(..., description="Collection name")
    documents: List[Dict[str, Any]] = Field(..., description="Documents to insert")


@app.post("/api/mongodb/query", dependencies=[Depends(get_api_key)])
async def query_mongodb(query: MongoQuery):
    """Query MongoDB collections (requires API key)"""
    if not mongo_client:
        raise HTTPException(status_code=503, detail="MongoDB not available")

    try:
        db = mongo_client[query.database]
        collection = db[query.collection]

        cursor = (
            collection.find(query.filter, query.projection)
            .limit(query.limit)
            .skip(query.skip)
        )

        if query.sort:
            cursor = cursor.sort(list(query.sort.items()))

        documents = list(cursor)

        # Convert ObjectId to string
        for doc in documents:
            if "_id" in doc:
                doc["_id"] = str(doc["_id"])

        return {"count": len(documents), "documents": documents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/api/mongodb/insert", dependencies=[Depends(get_api_key)])
async def insert_mongodb(insert: MongoInsert):
    """Insert documents into MongoDB (requires API key)"""
    if not mongo_client:
        raise HTTPException(status_code=503, detail="MongoDB not available")

    try:
        db = mongo_client[insert.database]
        collection = db[insert.collection]

        result = collection.insert_many(insert.documents)

        return {
            "inserted_count": len(result.inserted_ids),
            "inserted_ids": [str(id) for id in result.inserted_ids],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insert failed: {str(e)}")


@app.get("/api/mongodb/collections", dependencies=[Depends(get_api_key)])
async def list_mongodb_collections(database: str = "gitquery"):
    """List all collections in a MongoDB database (requires API key)"""
    if not mongo_client:
        raise HTTPException(status_code=503, detail="MongoDB not available")

    try:
        db = mongo_client[database]
        collections = db.list_collection_names()
        return {"database": database, "collections": collections}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list collections: {str(e)}"
        )


# ============================================================================
# Redis Query Endpoints
# ============================================================================


@app.get("/api/redis/get/{key}", dependencies=[Depends(get_api_key)])
async def get_redis_key(key: str):
    """Get value from Redis by key (requires API key)"""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")

    try:
        value = redis_client.get(key)
        return {"key": key, "value": value}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Get failed: {str(e)}")


@app.post("/api/redis/set", dependencies=[Depends(get_api_key)])
async def set_redis_key(
    key: str = Body(...), value: str = Body(...), expire: Optional[int] = Body(None)
):
    """Set value in Redis (requires API key)"""
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


@app.get("/api/redis/keys", dependencies=[Depends(get_api_key)])
async def list_redis_keys(pattern: str = "*", limit: int = Query(default=100, le=1000)):
    """List Redis keys matching pattern (requires API key)"""
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


# ============================================================================
# Qdrant Query Endpoints
# ============================================================================


class QdrantQuery(BaseModel):
    collection: str = Field(..., description="Collection name")
    vector: List[float] = Field(..., description="Query vector")
    limit: int = Field(default=10, le=100, description="Number of results")
    score_threshold: Optional[float] = Field(
        default=None, description="Minimum score threshold"
    )


class QdrantInsert(BaseModel):
    collection: str = Field(..., description="Collection name")
    points: List[Dict[str, Any]] = Field(
        ..., description="Points to insert (id, vector, payload)"
    )


@app.post("/api/qdrant/search", dependencies=[Depends(get_api_key)])
async def search_qdrant(query: QdrantQuery):
    """Search Qdrant vector database (requires API key)"""
    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Qdrant not available")

    try:
        results = qdrant_client.search(
            collection_name=query.collection,
            query_vector=query.vector,
            limit=query.limit,
            score_threshold=query.score_threshold,
        )

        return {
            "collection": query.collection,
            "count": len(results),
            "results": [
                {"id": result.id, "score": result.score, "payload": result.payload}
                for result in results
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/api/qdrant/insert", dependencies=[Depends(get_api_key)])
async def insert_qdrant(insert: QdrantInsert):
    """Insert points into Qdrant (requires API key)"""
    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Qdrant not available")

    try:
        points = [
            PointStruct(
                id=point.get("id"),
                vector=point["vector"],
                payload=point.get("payload", {}),
            )
            for point in insert.points
        ]

        qdrant_client.upsert(collection_name=insert.collection, points=points)

        return {"collection": insert.collection, "inserted_count": len(points)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Insert failed: {str(e)}")


@app.get("/api/qdrant/collections", dependencies=[Depends(get_api_key)])
async def list_qdrant_collections():
    """List all Qdrant collections (requires API key)"""
    if not qdrant_client:
        raise HTTPException(status_code=503, detail="Qdrant not available")

    try:
        collections = qdrant_client.get_collections()
        return {
            "collections": [
                {
                    "name": col.name,
                    "vectors_count": col.vectors_count,
                    "points_count": col.points_count,
                }
                for col in collections.collections
            ]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list collections: {str(e)}"
        )


# ============================================================================
# Batch Operations
# ============================================================================


class BatchInsert(BaseModel):
    mongodb_data: Optional[List[MongoInsert]] = None
    qdrant_data: Optional[List[QdrantInsert]] = None
    redis_data: Optional[List[Dict[str, Any]]] = None


@app.post("/api/batch/insert", dependencies=[Depends(get_api_key)])
async def batch_insert(batch: BatchInsert):
    """Batch insert data into multiple databases (requires API key)"""
    results = {"mongodb": [], "qdrant": [], "redis": []}
    errors = []

    # MongoDB batch insert
    if batch.mongodb_data:
        for insert_op in batch.mongodb_data:
            try:
                db = mongo_client[insert_op.database]
                collection = db[insert_op.collection]
                result = collection.insert_many(insert_op.documents)
                results["mongodb"].append(
                    {
                        "collection": insert_op.collection,
                        "inserted_count": len(result.inserted_ids),
                    }
                )
            except Exception as e:
                errors.append(f"MongoDB insert failed: {str(e)}")

    # Qdrant batch insert
    if batch.qdrant_data:
        for insert_op in batch.qdrant_data:
            try:
                points = [
                    PointStruct(
                        id=point.get("id"),
                        vector=point["vector"],
                        payload=point.get("payload", {}),
                    )
                    for point in insert_op.points
                ]
                qdrant_client.upsert(
                    collection_name=insert_op.collection, points=points
                )
                results["qdrant"].append(
                    {"collection": insert_op.collection, "inserted_count": len(points)}
                )
            except Exception as e:
                errors.append(f"Qdrant insert failed: {str(e)}")

    # Redis batch insert
    if batch.redis_data:
        for item in batch.redis_data:
            try:
                key = item["key"]
                value = item["value"]
                expire = item.get("expire")

                if expire:
                    redis_client.setex(key, expire, value)
                else:
                    redis_client.set(key, value)
                results["redis"].append({"key": key, "status": "success"})
            except Exception as e:
                errors.append(f"Redis insert failed: {str(e)}")

    return {"results": results, "errors": errors if errors else None}


# ============================================================================
# Documentation Endpoint
# ============================================================================


@app.get("/api/docs/examples")
async def get_examples():
    """Get example queries and usage patterns"""
    return {
        "mongodb_query": {
            "endpoint": "POST /api/mongodb/query",
            "example": {
                "collection": "users",
                "filter": {"username": "johndoe"},
                "limit": 10,
            },
        },
        "mongodb_insert": {
            "endpoint": "POST /api/mongodb/insert",
            "auth": "X-API-Key header required",
            "example": {
                "collection": "users",
                "documents": [
                    {"username": "alice", "email": "alice@example.com"},
                    {"username": "bob", "email": "bob@example.com"},
                ],
            },
        },
        "qdrant_search": {
            "endpoint": "POST /api/qdrant/search",
            "example": {
                "collection": "repository_embeddings",
                "vector": [0.1] * 768,
                "limit": 5,
            },
        },
        "qdrant_insert": {
            "endpoint": "POST /api/qdrant/insert",
            "auth": "X-API-Key header required",
            "example": {
                "collection": "repository_embeddings",
                "points": [
                    {
                        "id": 1,
                        "vector": [0.1] * 768,
                        "payload": {"repo_name": "example/repo"},
                    }
                ],
            },
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
