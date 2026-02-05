# 📚 Database Quick Reference Card

## 🚀 Quick Commands

### Development (Local Access)
```bash
cd DB
docker-compose -f docker-compose.all.yml up -d
```
**Access:** `localhost:27017` | `localhost:8081` | `localhost:6333`

### Production (Secure Network)
```bash
cd DB
docker-compose -f docker-compose.production.yml up -d
```
**Access:** `gitquery-mongodb:27017` | `gitquery-cosmos-db:8081` | `gitquery-qdrant:6333`

---

## 🔌 Connection Strings

### MongoDB
```python
# Development
mongodb://admin:pass@localhost:27017/gitquery?authSource=admin

# Production
mongodb://admin:pass@gitquery-mongodb:27017/gitquery?authSource=admin
```

### Cosmos DB
```python
# Development
https://localhost:8081

# Production
https://gitquery-cosmos-db:8081

# Emulator Key
C2y6yDjf5/R+ob0N8A7Cgv30VRDJIWEHLM+4QDU5DE2nQ9nDuVTqobD4b8mGGyPMbIZnqyMsEcaGQy67XIw/Jw==
```

### Qdrant
```python
# Development
http://localhost:6333

# Production
http://gitquery-qdrant:6333
```

---

## 🐍 Python Usage

### Quick Import
```python
from DB.database import db_manager

# MongoDB
mongo = db_manager.get_mongodb()
users = mongo.users.find()

# Cosmos DB
cosmos = db_manager.get_cosmos()
repos = cosmos.repositories.find()

# Qdrant
qdrant = db_manager.get_qdrant()
results = qdrant.search(collection_name="repository_embeddings", ...)
```

### Direct Import
```python
from DB.db_config import get_mongodb_db, get_cosmos_db, get_qdrant_client

mongo = get_mongodb_db()
cosmos = get_cosmos_db()
qdrant = get_qdrant_client()
```

---

## 📊 Collections

### MongoDB
- `users` - User data
- `sessions` - Active sessions
- `interactions` - Query history
- `recommendations` - Generated recommendations

### Cosmos DB
- `repositories` - GitHub repo metadata
- `repository_activity` - Repo events

### Qdrant
- `repository_embeddings` - Repo vectors (768D)
- `code_embeddings` - Code vectors (768D)
- `readme_embeddings` - Doc vectors (768D)

---

## 🛠️ Management

### Status
```bash
docker ps --filter name=gitquery
```

### Logs
```bash
docker logs -f gitquery-mongodb
docker logs -f gitquery-cosmos-db
docker logs -f gitquery-qdrant
```

### Stop
```bash
docker-compose -f docker-compose.all.yml down
```

### Clean (⚠️ Deletes Data)
```bash
docker-compose -f docker-compose.all.yml down -v
```

---

## 🔐 Environment Variables

### Required
```bash
MONGO_USER=admin
MONGO_PASSWORD=your_secure_password
QDRANT_API_KEY=your_api_key
```

### Optional
```bash
MONGO_DB=gitquery
COSMOS_PARTITION_COUNT=10
QDRANT_LOG_LEVEL=INFO
```

---

## 📖 Documentation

| File | Content |
|------|---------|
| **README.md** | Main overview |
| **DATABASE_ACCESS_GUIDE.md** | In/Out access methods |
| **SECURITY.md** | Security configuration |
| **PIPELINE_DEVELOPER_GUIDE.md** | Pipeline development |

---

## 🔍 Troubleshooting

### Can't connect
```bash
# Check containers
docker ps --filter name=gitquery

# Check logs
docker logs gitquery-mongodb

# Check network
docker network inspect gitquery-db-network
```

### Port in use
```bash
# Windows
netstat -ano | findstr :27017
taskkill /F /PID <pid>

# Or change port
ports:
  - "27018:27017"
```

### Auth failed
```bash
# Check env vars
docker exec gitquery-mongodb printenv | grep MONGO

# Try default
docker exec -it gitquery-mongodb mongosh -u admin -p mongopass
```

---

## 📦 Ports

| Database | HTTP | Other |
|----------|------|-------|
| MongoDB | 27017 | - |
| Cosmos DB | 8081 | 10250-10255 |
| Qdrant | 6333 | 6334 (gRPC) |

**Development:** Exposed to `127.0.0.1` only  
**Production:** Not exposed (internal network only)

---

## 🧪 Quick Test

```python
from pymongo import MongoClient
from qdrant_client import QdrantClient

# Test MongoDB
mongo = MongoClient("mongodb://admin:mongopass@localhost:27017/")
mongo.admin.command('ping')
print("✓ MongoDB OK")

# Test Qdrant
qdrant = QdrantClient(host="localhost", port=6333)
qdrant.get_collections()
print("✓ Qdrant OK")
```

---

## 🐳 Docker Network

**Name:** `gitquery-db-network`  
**Type:** Bridge

### Add your container:
```yaml
services:
  my-app:
    networks:
      - gitquery-db-network

networks:
  gitquery-db-network:
    external: true
```

---

**Full Docs:** See [README.md](README.md) and [DATABASE_ACCESS_GUIDE.md](DATABASE_ACCESS_GUIDE.md)
