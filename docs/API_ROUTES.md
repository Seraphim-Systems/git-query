# API Routes Documentation

Complete reference for all API endpoints in git-query.

## Base URLs

Current Port Allocation:

gateway: 8000
mcp-server: 8090
mongodb: 27017
redis: 6379
qdrant: 6333, 6334
nginx: 80
nginx: 80

- **Root:** `/` - Serves static web application
- **API:** `/api/` - API endpoints
- **Health:** `/api/health` - System health checks (public)
- **Database:** `/api/{service}/` - Database operations (requires API key)

---

## Authentication

### Public Endpoints (No Auth Required)
- `GET /api/health`
- `GET /api/health/databases`

### Protected Endpoints (API Key Required)
All `/api/*` endpoints require an API key in the `Authorization` header:

```http
Authorization: Bearer <your-api-key>
```

**API Keys (per service):**
- MongoDB: `APIKEY_MONGODB`
- Redis: `APIKEY_REDIS`
- Qdrant: `APIKEY_QDRANT`
- MCP: `APIKEY_MCP`

---

## Health Endpoints

### GET /api/health
Returns overall system health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-02-08T12:00:00Z",
  "services": {
    "gateway": true,
    "mongodb": true,
    "redis": true,
    "qdrant": false,
    "mcp_server": false
  }
}
```

**Status Codes:**
- `200` - At least one service healthy
- `503` - All services down

---

### GET /api/health/databases
Returns database-specific health checks.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-02-08T12:00:00Z",
  "databases": {
    "mongodb": {"status": true, "url": "mongodb://mongodb:27017"},
    "redis": {"status": true, "url": "redis://redis:6379"},
    "qdrant": {"status": false}
  }
}
```

---

## MongoDB Endpoints

Base path: `/api/mongodb`

### GET /api/mongodb/collections
List all collections in the database.

**Auth:** MongoDB API key required

**Response:**
```json
{
  "database": "gitquery",
  "collections": ["repositories", "commits", "users"]
}
```

---

### POST /api/mongodb/{collection}/query
Query documents in a collection.

**Auth:** MongoDB API key required

**Request Body:**
```json
{
  "filter": {"stars": {"$gt": 100}},
  "projection": {"name": 1, "stars": 1},
  "limit": 10,
  "skip": 0,
  "sort": {"stars": -1}
}
```

**Response:**
```json
{
  "count": 10,
  "documents": [
    {"_id": "123", "name": "repo1", "stars": 500}
  ]
}
```

---

### POST /api/mongodb/{collection}/bulk
Bulk insert or upsert documents.

**Auth:** MongoDB API key required

**Request Body:**
```json
{
  "documents": [
    {"_id": "1", "name": "repo1", "stars": 100},
    {"_id": "2", "name": "repo2", "stars": 200}
  ],
  "ordered": false,
  "upsert": true
}
```

**Response:**
```json
{
  "inserted": 0,
  "updated": 2,
  "errors": []
}
```

**Use Cases:**
- Loading complete datasets from JSON files
- Batch updates from pipelines
- Idempotent data ingestion (use `upsert: true`)

---

## Redis Endpoints

Base path: `/api/redis`

### GET /api/redis/{key}
Get value for a key.

**Auth:** Redis API key required

**Response:**
```json
{
  "key": "cache:user:123",
  "value": "...",
  "ttl": 3600
}
```

---

### PUT /api/redis/{key}
Set value for a key with optional TTL.

**Auth:** Redis API key required

**Request Body:**
```json
{
  "value": "data",
  "ttl": 3600
}
```

**Response:**
```json
{
  "key": "cache:user:123",
  "status": "success"
}
```

---

### POST /api/redis/batch
Execute batch Redis operations.

**Auth:** Redis API key required

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
    {"key": "k2", "value": "v2", "status": "ok"},
    {"key": "k3", "status": "deleted"}
  ]
}
```

**Use Cases:**
- Bulk cache invalidation
- Loading cache from pipelines
- Batch cache operations

---

## Qdrant Endpoints

Base path: `/api/qdrant`

### GET /api/qdrant/collections
List all vector collections.

**Auth:** Qdrant API key required

**Response:**
```json
{
  "collections": [
    {
      "name": "embeddings",
      "vectors_count": 1000,
      "points_count": 1000
    }
  ]
}
```

---

### POST /api/qdrant/{collection}/search
Search for similar vectors.

**Auth:** Qdrant API key required

**Request Body:**
```json
{
  "vector": [0.1, 0.2, 0.3, ...],
  "limit": 10,
  "filter": {},
  "with_payload": true
}
```

**Response:**
```json
{
  "collection": "embeddings",
  "count": 10,
  "results": [
    {
      "id": "vec1",
      "score": 0.95,
      "payload": {"name": "item1"}
    }
  ]
}
```

---

### POST /api/qdrant/{collection}/bulk
Bulk upsert vectors.

**Auth:** Qdrant API key required

**Request Body:**
```json
{
  "points": [
    {"id": "1", "vector": [0.1, 0.2], "payload": {"name": "item1"}},
    {"id": "2", "vector": [0.3, 0.4], "payload": {"name": "item2"}}
  ],
  "wait": true
}
```

**Response:**
```json
{
  "collection": "embeddings",
  "upserted": 2,
  "status": "completed"
}
```

**Use Cases:**
- Loading pre-computed embeddings
- Batch vector ingestion from ML pipelines
- Updating vector database from training jobs

---

## MCP Endpoints

Base path: `/api/mcp`

### GET /api/mcp/tools
List available MCP tools.

**Auth:** MCP API key required

**Response:**
```json
{
  "tools": [
    {"name": "recommend", "description": "Get recommendations"},
    {"name": "example_tool", "description": "Example tool"}
  ]
}
```

---

### POST /api/mcp/tools/{tool_name}
Execute an MCP tool (recommendation engine, etc.).

**Auth:** MCP API key required

**Request Body:**
```json
{
  "user_id": "123",
  "context": {...}
}
```

**Response:** Tool-specific

---

## Error Responses

All endpoints return consistent error format:

```json
{
  "detail": "Error message here"
}
```

**Common Status Codes:**
- `200` - Success
- `400` - Bad request (invalid payload)
- `401` - Unauthorized (missing/invalid API key)
- `403` - Forbidden (wrong API key for service)
- `404` - Not found (collection/key doesn't exist)
- `500` - Internal server error
- `503` - Service unavailable (database down)

---

## Rate Limiting

- Default: 100 requests per minute per API key
- Bulk operations: Same rate limit (operations counted as 1 request)
- Health endpoints: No rate limiting

Exceeded rate limit returns `429 Too Many Requests`.

---

## Example Usage

### Python
```python
import requests

API_KEY = "your-mongodb-api-key"
BASE_URL = "https://api.gitquery.com/api"

headers = {"Authorization": f"Bearer {API_KEY}"}

# Bulk insert
response = requests.post(
    f"{BASE_URL}/mongodb/repositories/bulk",
    json={
        "documents": [
            {"_id": "1", "name": "repo1", "stars": 100},
            {"_id": "2", "name": "repo2", "stars": 200}
        ],
        "upsert": True
    },
    headers=headers
)
print(response.json())
```

### cURL
```bash
# Query MongoDB
curl -X POST "https://api.gitquery.com/api/mongodb/repositories/query" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"filter": {"stars": {"$gt": 100}}, "limit": 10}'

# Bulk upsert to Qdrant
curl -X POST "https://api.gitquery.com/api/qdrant/embeddings/bulk" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"points": [{"id": "1", "vector": [0.1, 0.2], "payload": {"name": "item1"}}]}'
```

---

## Interactive Documentation

Visit `/docs` (Swagger UI) or `/redoc` (ReDoc) for interactive API documentation and testing.
