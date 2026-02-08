# Docker Compose Stack Architecture

This directory contains modularized Docker Compose files for deploying the git-query infrastructure.

## Architecture Overview

The compose files are split into service categories, each independently deployable:

**Infrastructure Layer:**
- **docker-compose.base.yml** - Base infrastructure (networks & volumes only, no containers)
- **docker-compose.mongodb.yml** - MongoDB database
- **docker-compose.cosmos.yml** - Cosmos DB emulator
- **docker-compose.qdrant.yml** - Qdrant vector database
- **docker-compose.redis.yml** - Redis cache
- **docker-compose.db-api.yml** - Database Query API
- **docker-compose.nginx.yml** - Nginx reverse proxy

**Data Processing Layer:**
- **docker-compose.pipelines.yml** - Data pipelines (scraper, processing)

**ML/AI Layer:**
- **docker-compose.mcp.yml** - MCP (Model Context Protocol) servers
- **docker-compose.reco.yml** - Recommendation engine (serving)
- **docker-compose.training.yml** - ML model training (batch job)

## Usage

### Deploy Infrastructure Services

```bash
# Deploy databases
docker-compose -f docker-compose.base.yml -f docker-compose.mongodb.yml up -d
docker-compose -f docker-compose.base.yml -f docker-compose.qdrant.yml up -d
docker-compose -f docker-compose.base.yml -f docker-compose.redis.yml up -d

# Deploy DB API (after databases are ready)
docker-compose -f docker-compose.base.yml -f docker-compose.db-api.yml up -d

# Deploy nginx (after DB API is ready)
docker-compose -f docker-compose.base.yml -f docker-compose.nginx.yml up -d
```

### Deploy Data Processing Services

```bash
# Deploy pipelines (requires databases)
docker-compose -f docker-compose.base.yml -f docker-compose.pipelines.yml up -d
```

### Deploy ML/AI Services

```bash
# Deploy MCP servers (requires databases)
docker-compose -f docker-compose.base.yml -f docker-compose.mcp.yml up -d

# Deploy recommendation engine (requires databases)
docker-compose -f docker-compose.base.yml -f docker-compose.reco.yml up -d

# Run training job (one-time execution)
docker-compose -f docker-compose.base.yml -f docker-compose.training.yml up --build
```

### Deploy Full Stack

Deploy everything:

```bash
docker-compose -f docker-compose.base.yml \
  -f docker-compose.mongodb.yml \
  -f docker-compose.cosmos.yml \
  -f docker-compose.qdrant.yml \
  -f docker-compose.redis.yml \
  -f docker-compose.db-api.yml \
  -f docker-compose.nginx.yml \
  up -d --build
```

### Stop Services

```bash
# Stop individual service
docker-compose -f docker-compose.base.yml -f docker-compose.mongodb.yml down

# Stop all git-query services
docker ps --filter "name=git-query-" -q | xargs docker stop
```

## CI/CD Workflows

Each service category has its own deployment workflow (manual dispatch only):

**Data & ML Workflows:**
- `.github/workflows/deploy-pipelines.yml` - Deploy data pipelines (scraper, processing)
- `.github/workflows/deploy-mcp.yml` - Deploy MCP servers
- `.github/workflows/deploy-reco.yml` - Deploy recommendation engine
- `.github/workflows/deploy-training.yml` - Run ML training (also scheduled weekly)

**Production Deployment:**
- `.github/workflows/deploy-prod.yml` - Deploy full production stack (legacy)

### Workflow Triggers

**All workflows use manual dispatch only** - no automatic deployments on push.

To deploy:
1. Go to GitHub Actions
2. Select the workflow (deploy-pipelines, deploy-mcp, deploy-reco, or deploy-training)
3. Click "Run workflow"
4. Choose environment (dev/prod) and service-specific options
5. Confirm deployment

**Exception:** Training workflow also runs automatically every Sunday at 2 AM UTC (scheduled).

## Dependencies

Service dependencies by category:

**Infrastructure:**
- **MongoDB, Cosmos, Qdrant, Redis**: Independent, can be deployed in any order
- **DB API**: Requires MongoDB, Redis, and Qdrant to be running
- **Nginx**: Requires DB API to be running

**Data Processing:**
- **Scraper**: Requires MongoDB and Redis
- **Processing**: Requires MongoDB, Qdrant, and Redis

**ML/AI:**
- **MCP Server**: Requires MongoDB, Qdrant, and Redis
- **Recommender**: Requires MongoDB, Qdrant, and Redis
- **Training**: Requires MongoDB and Qdrant (batch job)

### Deployment Order

For full stack deployment:
1. Deploy databases (MongoDB, Qdrant, Redis, Cosmos)
2. Deploy DB API
3. Deploy pipelines (scraper, processing)
4. Deploy ML services (MCP, recommender)
5. Deploy nginx gateway
6. Run training (optional, as needed)

## Environment Variables

Create a `.env` file with the new secret naming convention:

```bash
# Database Credentials
DB_MONGODB_USER=admin
DB_MONGODB_PASSWORD=<password>
DB_MONGODB_DATABASE=gitquery
DB_REDIS_PASSWORD=<password>
DB_COSMOS_PARTITION_COUNT=10
DB_QDRANT_API_KEY=<api-key>

# API Keys (per-service authentication)
APIKEY_MONGODB=<mongodb-api-key>
APIKEY_REDIS=<redis-api-key>
APIKEY_QDRANT=<qdrant-api-key>
APIKEY_MCP=<mcp-api-key>

# Application Secrets
APP_JWT_SECRET=<jwt-secret>
APP_SESSION_SECRET=<session-secret>
APP_OPENAI_API_KEY=<openai-key>

# Service Configuration
SVC_NGINX_SERVER_NAME=<your-domain>
SVC_QDRANT_LOG_LEVEL=INFO
SVC_LOG_LEVEL=INFO

# Pipelines (optional)
SCRAPE_INTERVAL=3600
MAX_CONCURRENT_JOBS=5
BATCH_SIZE=100

# ML/Recommender (optional)
INFERENCE_BATCH_SIZE=32
TRAINING_BATCH_SIZE=64
EPOCHS=10
LEARNING_RATE=0.001
```

**Note:** The new naming convention follows the pattern `<CATEGORY>_<SERVICE>_<PURPOSE>` for better organization:
- `DB_*` - Database credentials
- `APIKEY_*` - API keys for service endpoints
- `APP_*` - Application-level secrets
- `SVC_*` - Service configuration

## Legacy Files

The following files are kept for backward compatibility:

- **docker-compose.dev.yml** - Development overrides (exposes ports)
- **docker-compose.prod.yml** - Production overrides (secrets, restart policies)
- **.github/workflows/deploy-prod.yml** - Legacy production deployment (uses docker-compose.db.yml)

**Note:** docker-compose.db.yml is the active infrastructure compose file, not legacy.
- docker-compose.base.yml + docker-compose.training.yml
