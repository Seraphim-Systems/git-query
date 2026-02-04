# Git-Query Database Infrastructure

Complete multi-database setup for the Git-Query GitHub repository recommendation chatbot.

## 🗄️ Database Architecture

```
┌──────────────────────────────────────────────────────────┐
│              Git-Query Application Layer                  │
└──────────────────────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Cosmos DB   │  │   MongoDB    │  │   Qdrant     │
│              │  │              │  │              │
│ Repository   │  │ Users        │  │ Vector       │
│ Metadata     │  │ Sessions     │  │ Embeddings   │
│ (Large-scale)│  │ Interactions │  │ (Semantic    │
│              │  │              │  │  Search)     │
└──────────────┘  └──────────────┘  └──────────────┘
        │                │                │
        └────────────────┼────────────────┘
                         │
                         ▼
                 ┌──────────────┐
                 │    Redis     │
                 │              │
                 │ Cache Layer  │
                 │ (Fast Access)│
                 └──────────────┘
```

## 📊 Database Roles

| Database | Purpose | Data Type | Scale |
|----------|---------|-----------|-------|
| **Azure Cosmos DB** | Large-scale repository data | Repository metadata, activity logs | Millions of repos |
| **MongoDB** | General application data | User profiles, chat sessions, interactions | Thousands of users |
| **Qdrant** | Vector search | Embeddings for semantic search | Millions of vectors |
| **Redis** | High-speed cache | API results, sessions, rate limiting | Sub-millisecond access |
| **PostgreSQL** | Legacy/compatibility | Structured relational data | Optional |

## 🚀 Quick Start

### 1. Prerequisites

- Docker Desktop (Windows) or Docker + Docker Compose (Linux/Mac)
- 8GB+ RAM available
- 20GB+ disk space

### 2. Clone and Configure

```powershell
# Navigate to DB directory
cd DB

# Copy environment template
cp DB_Deploy/.env.example .env

# (Optional) Edit .env - defaults work for local development
```

### 3. Start All Databases

```powershell
# Start all databases with one command
docker-compose -f docker-compose.all.yml up -d

# Or start individually (see below)
```

### 4. Initialize Qdrant

```powershell
# Initialize Qdrant vector collections
docker-compose -f docker-compose.qdrant.yml up qdrant-init
```

### 5. Verify Health

```powershell
# Windows PowerShell
.\DB_Deploy\health-check.ps1

# Linux/Mac
chmod +x DB_Deploy/health-check.sh
./DB_Deploy/health-check.sh
```

## 📦 Individual Database Management

### Start Individual Databases

```powershell
# MongoDB only
docker-compose -f docker-compose.mongodb.yml up -d

# Cosmos DB only (requires 3GB RAM)
docker-compose -f docker-compose.cosmos.yml up -d

# Qdrant only
docker-compose -f docker-compose.qdrant.yml up -d

# Redis only
docker-compose -f docker-compose.redis.yml up -d

# PostgreSQL only
docker-compose -f docker-compose.postgres.yml up -d
```

### Stop Individual Databases

```powershell
docker-compose -f docker-compose.[database].yml down
```

## 🔌 Connection Strings (Local Development)

```bash
# MongoDB
mongodb://admin:mongopass@localhost:27017/gitquery?authSource=admin

# Cosmos DB Emulator
https://localhost:8081
Key: C2y6yDjf5/R+ob0N8A7Cgv30VRDJIWEHLM+4QDU5DE2nQ9nDuVTqobD4b8mGGyPMbIZnqyMsEcaGQy67XIw/Jw==

# Qdrant
http://localhost:6333

# Redis
redis://localhost:6379

# PostgreSQL
postgresql://user:pass@localhost:5432/gitquery
```

## 💻 Using the Database Client

### Python Client

```python
from DB.db_config import db_clients, get_mongodb_db, get_qdrant_client

# MongoDB - Store user session
db = get_mongodb_db()
db.users.insert_one({
    'user_id': 'user_123',
    'username': 'john_doe',
    'created_at': datetime.utcnow()
})

# Qdrant - Search similar repositories
qdrant = get_qdrant_client()
results = qdrant.search(
    collection_name="repository_embeddings",
    query_vector=embedding,
    limit=10
)

# Redis - Cache data
redis = db_clients.redis
redis.setex('key', 3600, 'value')  # 1 hour cache

# Cosmos DB - Store repository data
cosmos = db_clients.cosmos['gitquery_cosmos']
cosmos.repositories.insert_one({
    'repo_id': 'repo_123',
    'full_name': 'tensorflow/tensorflow',
    'stars': 150000
})
```

## 📂 Directory Structure

```
DB/
├── docker-compose.all.yml        # Master compose file (all databases)
├── docker-compose.mongodb.yml    # MongoDB only
├── docker-compose.cosmos.yml     # Cosmos DB only
├── docker-compose.qdrant.yml     # Qdrant only
├── docker-compose.redis.yml      # Redis only
├── docker-compose.postgres.yml   # PostgreSQL only
├── db_config.py                  # Python database client
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── mongodb/
│   └── mongo-init.js            # MongoDB initialization
│
├── cosmos/
│   └── cosmos-init.js           # Cosmos DB initialization
│
├── qdrant/
│   ├── qdrant-init.py          # Qdrant initialization script
│   └── Dockerfile.qdrant-init  # Qdrant init container
│
├── redis/
│   └── redis.conf              # Redis configuration
│
├── postgres/
│   └── init-db.sql             # PostgreSQL initialization
│
└── DB_Deploy/                   # Deployment files only
    ├── .env.example            # Environment variables template
    ├── deployment-guide.md     # Full deployment guide
    ├── health-check.ps1        # Health check (Windows)
    └── health-check.sh         # Health check (Linux/Mac)
```

## 🔧 Common Commands

### View Logs

```powershell
# All databases
docker-compose -f docker-compose.all.yml logs -f

# Specific database
docker logs -f gitquery-mongodb
docker logs -f gitquery-qdrant
docker logs -f gitquery-redis
```

### Restart Services

```powershell
# Restart all
docker-compose -f docker-compose.all.yml restart

# Restart specific
docker restart gitquery-mongodb
```

### Stop All Databases

```powershell
docker-compose -f docker-compose.all.yml down
```

### Remove All Data (Clean Slate)

```powershell
# WARNING: This deletes all database data!
docker-compose -f docker-compose.all.yml down -v
```

## 📊 Data Models

### MongoDB Collections

- **users**: User profiles and preferences
- **chat_sessions**: Conversation history
- **user_interactions**: User actions (views, stars, forks)
- **recommendations**: Recommendation history with scores

### Cosmos DB Collections

- **repositories**: GitHub repository metadata
- **repository_activity**: Activity logs (commits, releases, issues)

### Qdrant Collections

- **repository_embeddings**: Vector embeddings for repo descriptions
- **code_embeddings**: Vector embeddings for code snippets
- **user_preference_embeddings**: Vector embeddings for user preferences

### Redis Cache Keys

- `repo:{repo_id}`: Repository metadata (TTL: 1 hour)
- `user:{user_id}:session`: User session data (TTL: 24 hours)
- `ratelimit:{user_id}`: API rate limiting (TTL: 1 minute)
- `query:{hash}`: Query results (TTL: 15 minutes)

## 🌐 Production Deployment

For production, use managed cloud services:

### Azure
- **Cosmos DB**: Azure Cosmos DB (MongoDB API)
- **MongoDB**: MongoDB Atlas on Azure
- **Qdrant**: Qdrant Cloud or self-hosted on AKS
- **Redis**: Azure Cache for Redis

### AWS
- **Cosmos DB**: Amazon DocumentDB
- **MongoDB**: MongoDB Atlas on AWS
- **Qdrant**: Self-hosted on EKS
- **Redis**: Amazon ElastiCache

### GCP
- **Cosmos DB**: Cloud Firestore or MongoDB Atlas
- **MongoDB**: MongoDB Atlas on GCP
- **Qdrant**: Self-hosted on GKE
- **Redis**: Cloud Memorystore

**See**: [DB_Deploy/deployment-guide.md](DB_Deploy/deployment-guide.md) for full production setup.

## 🔒 Security Best Practices

1. **Change default passwords** in `.env`
2. **Enable authentication** on all databases
3. **Use SSL/TLS** for all connections in production
4. **Restrict network access** with firewall rules
5. **Regular backups** to secure storage
6. **Rotate credentials** periodically
7. **Monitor access logs** for suspicious activity
8. **Use secrets management** (Azure Key Vault, AWS Secrets Manager)

## 🐛 Troubleshooting

### Port Conflicts

Edit `.env` to change ports:
```bash
MONGO_PORT=27018
REDIS_PORT=6380
```

### Container Won't Start

```powershell
# Check logs
docker logs gitquery-mongodb

# Check disk space
docker system df

# Clean up
docker system prune
```

### Connection Issues

```powershell
# Verify containers running
docker ps --filter "name=gitquery-"

# Check network
docker network ls | findstr gitquery

# Test from inside container
docker exec -it gitquery-mongodb mongosh
```

### Cosmos DB Certificate Issues

```powershell
# Download certificate
curl -k https://localhost:8081/_explorer/emulator.pem -o emulator.pem

# Trust certificate (Windows)
certutil -addstore "Root" emulator.pem
```

## 📚 Additional Resources

- **Deployment Guide**: [DB_Deploy/deployment-guide.md](DB_Deploy/deployment-guide.md)
- **Environment Template**: [DB_Deploy/.env.example](DB_Deploy/.env.example)
- **Health Check Scripts**: [DB_Deploy/health-check.*](DB_Deploy/)

### Official Documentation

- [MongoDB Documentation](https://docs.mongodb.com/)
- [Azure Cosmos DB Documentation](https://learn.microsoft.com/azure/cosmos-db/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [Redis Documentation](https://redis.io/documentation)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)

## 📝 License

Part of the Git-Query project. See main repository for license information.

---

**Last Updated**: February 2026  
**Version**: 1.0.0
