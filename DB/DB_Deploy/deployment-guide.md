# Database Deployment Guide

Complete guide for deploying Git-Query databases in various environments.

## Table of Contents

- [Quick Start (Local Development)](#quick-start-local-development)
- [Individual Database Deployment](#individual-database-deployment)
- [Production Deployment](#production-deployment)
- [Cloud Services Setup](#cloud-services-setup)
- [Troubleshooting](#troubleshooting)

---

## Quick Start (Local Development)

### Prerequisites

- Docker Desktop installed and running
- Docker Compose v2.0+
- 8GB+ RAM available
- 20GB+ disk space

### 1. Clone and Navigate

```powershell
cd DB
```

### 2. Configure Environment

```powershell
# Copy environment template
cp DB_Deploy/.env.example .env

# Edit .env with your settings (defaults work for local dev)
notepad .env
```

### 3. Start All Databases

```powershell
# Start all databases at once
docker-compose -f docker-compose.all.yml up -d

# Or start without Cosmos DB (saves resources)
docker-compose -f docker-compose.mongodb.yml up -d
docker-compose -f docker-compose.qdrant.yml up -d
docker-compose -f docker-compose.redis.yml up -d
docker-compose -f docker-compose.postgres.yml up -d
```

### 4. Initialize Qdrant

```powershell
# Wait for Qdrant to be ready, then initialize
docker-compose -f docker-compose.qdrant.yml up qdrant-init
```

### 5. Verify Health

```powershell
# Run health check
.\DB_Deploy\health-check.ps1

# Or check manually
docker ps --filter "name=gitquery-"
```

---

## Individual Database Deployment

### MongoDB Only

```powershell
# Start MongoDB
docker-compose -f docker-compose.mongodb.yml up -d

# Check status
docker logs gitquery-mongodb

# Connect
docker exec -it gitquery-mongodb mongosh -u admin -p mongopass
```

**Use Case**: User profiles, chat sessions, interactions

### Cosmos DB Only

```powershell
# Start Cosmos DB Emulator (requires 3GB RAM)
docker-compose -f docker-compose.cosmos.yml up -d

# Wait for initialization (can take 2-3 minutes)
docker logs -f gitquery-cosmos-db

# Access UI: https://localhost:8081/_explorer/index.html
```

**Use Case**: Large-scale repository metadata and activity

### Qdrant Only

```powershell
# Start Qdrant
docker-compose -f docker-compose.qdrant.yml up -d

# Initialize collections
docker-compose -f docker-compose.qdrant.yml up qdrant-init

# Check collections
curl http://localhost:6333/collections
```

**Use Case**: Vector embeddings for semantic search

### Redis Only

```powershell
# Start Redis
docker-compose -f docker-compose.redis.yml up -d

# Test connection
docker exec gitquery-redis redis-cli ping
```

**Use Case**: Caching, rate limiting, session storage

### PostgreSQL Only

```powershell
# Start PostgreSQL
docker-compose -f docker-compose.postgres.yml up -d

# Check status
docker exec gitquery-postgres pg_isready -U user
```

**Use Case**: Legacy/compatibility, structured data

---

## Production Deployment

### Azure Cloud Setup

#### 1. Azure Cosmos DB

```bash
# Create Cosmos DB account (MongoDB API)
az cosmosdb create \
  --name gitquery-cosmos \
  --resource-group gitquery-rg \
  --kind MongoDB \
  --capabilities EnableServerless EnableMongo

# Get connection string
az cosmosdb keys list \
  --name gitquery-cosmos \
  --resource-group gitquery-rg \
  --type connection-strings
```

**Update .env**:
```bash
COSMOS_DB_URL=mongodb://gitquery-cosmos.mongo.cosmos.azure.com:10255/
COSMOS_DB_KEY=your-primary-key-here
```

#### 2. MongoDB Atlas

```bash
# Sign up at https://www.mongodb.com/cloud/atlas
# Create cluster (M10+ for production)
# Whitelist IP addresses
# Create database user
```

**Update .env**:
```bash
MONGODB_URL=mongodb+srv://user:pass@cluster.mongodb.net/gitquery?retryWrites=true&w=majority
```

#### 3. Qdrant Cloud

```bash
# Sign up at https://cloud.qdrant.io
# Create cluster
# Copy cluster URL and API key
```

**Update .env**:
```bash
QDRANT_URL=https://your-cluster-id.qdrant.io
QDRANT_API_KEY=your-api-key-here
```

#### 4. Azure Cache for Redis

```bash
# Create Redis cache
az redis create \
  --name gitquery-redis \
  --resource-group gitquery-rg \
  --location eastus \
  --sku Premium \
  --vm-size P1

# Get access key
az redis list-keys \
  --name gitquery-redis \
  --resource-group gitquery-rg
```

**Update .env**:
```bash
REDIS_URL=rediss://gitquery-redis.redis.cache.windows.net:6380
REDIS_PASSWORD=your-access-key-here
```

### Production Docker Compose

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  # Production uses cloud services, no local containers needed
  # This file is for application services that connect to cloud DBs

  app:
    environment:
      - MONGODB_URL=${MONGODB_URL}
      - COSMOS_DB_URL=${COSMOS_DB_URL}
      - COSMOS_DB_KEY=${COSMOS_DB_KEY}
      - QDRANT_URL=${QDRANT_URL}
      - QDRANT_API_KEY=${QDRANT_API_KEY}
      - REDIS_URL=${REDIS_URL}
      - REDIS_PASSWORD=${REDIS_PASSWORD}
```

---

## Cloud Services Setup

### AWS Alternatives

- **MongoDB**: Amazon DocumentDB
- **Redis**: Amazon ElastiCache for Redis
- **Postgres**: Amazon RDS for PostgreSQL
- **Vector DB**: Self-hosted Qdrant on EC2 or EKS

### GCP Alternatives

- **MongoDB**: MongoDB Atlas on GCP
- **Redis**: Google Cloud Memorystore for Redis
- **Postgres**: Cloud SQL for PostgreSQL
- **Vector DB**: Self-hosted Qdrant on GCE or GKE

---

## Maintenance Commands

### Backup

```powershell
# MongoDB backup
docker exec gitquery-mongodb mongodump --out=/backup

# Redis backup
docker exec gitquery-redis redis-cli SAVE

# PostgreSQL backup
docker exec gitquery-postgres pg_dump -U user gitquery > backup.sql
```

### Restore

```powershell
# MongoDB restore
docker exec gitquery-mongodb mongorestore /backup

# Redis restore
docker cp dump.rdb gitquery-redis:/data/

# PostgreSQL restore
docker exec -i gitquery-postgres psql -U user gitquery < backup.sql
```

### Update

```powershell
# Pull latest images
docker-compose -f docker-compose.all.yml pull

# Restart with new images
docker-compose -f docker-compose.all.yml up -d
```

### Clean Up

```powershell
# Stop all databases
docker-compose -f docker-compose.all.yml down

# Remove volumes (DELETES ALL DATA)
docker-compose -f docker-compose.all.yml down -v

# Remove everything including images
docker-compose -f docker-compose.all.yml down -v --rmi all
```

---

## Troubleshooting

### Port Already in Use

```powershell
# Check what's using the port
netstat -ano | findstr :27017

# Kill the process (replace PID)
taskkill /PID <PID> /F

# Or change port in .env
MONGO_PORT=27018
```

### Container Won't Start

```powershell
# Check logs
docker logs gitquery-mongodb

# Check disk space
docker system df

# Clean up unused resources
docker system prune -a
```

### Can't Connect to Database

```powershell
# Verify container is running
docker ps --filter "name=gitquery-"

# Check network
docker network ls | findstr gitquery

# Test connection from inside container
docker exec -it gitquery-mongodb mongosh
```

### Cosmos DB Emulator Issues

```powershell
# Cosmos emulator needs 3GB RAM minimum
# Accept self-signed certificate
# https://localhost:8081/_explorer/emulator.pem

# Import certificate to Windows
certutil -addstore "Root" emulator.pem
```

### Qdrant Not Initializing

```powershell
# Check if Qdrant is healthy first
curl http://localhost:6333/readyz

# Run initialization manually
docker-compose -f docker-compose.qdrant.yml up qdrant-init

# Check initialization logs
docker logs gitquery-qdrant-init
```

---

## Performance Tuning

### MongoDB

- Enable profiling: `db.setProfilingLevel(1, { slowms: 100 })`
- Add indexes for frequently queried fields
- Use connection pooling (default in drivers)

### Redis

- Adjust maxmemory based on usage
- Use pipelining for bulk operations
- Enable AOF + RDB for persistence

### Qdrant

- Adjust `indexing_threshold` for collection size
- Use HNSW parameters for speed/accuracy tradeoff
- Enable quantization for memory savings

### PostgreSQL

- Configure shared_buffers (25% of RAM)
- Enable query logging for optimization
- Use connection pooling (pgBouncer)

---

## Security Checklist

- [ ] Change all default passwords
- [ ] Enable authentication on all databases
- [ ] Use strong passwords (16+ characters)
- [ ] Enable SSL/TLS for all connections
- [ ] Restrict network access (firewall rules)
- [ ] Regular backups to secure storage
- [ ] Rotate credentials periodically
- [ ] Monitor access logs
- [ ] Use secrets management (Azure Key Vault, AWS Secrets Manager)
- [ ] Enable audit logging in production

---

## Next Steps

1. ✅ Databases deployed and configured
2. Configure application to use databases
3. Load initial data / seed databases
4. Set up monitoring and alerts
5. Configure automated backups
6. Run load testing
7. Deploy to production

---

## Support

- **Issues**: Create an issue in the repository
- **Documentation**: See main README.md
- **Health Check**: Run `.\DB_Deploy\health-check.ps1`

---

**Last Updated**: February 2026
