# Pipelines

This directory contains data pipeline scripts for the Git-Query chatbot system.

## Pipeline Components

### 1. **GitHub Scraper** (`github_scraper.py`)
Scrapes repository data from GitHub API and stores raw data in Cosmos DB.

**Features:**
- Collects comprehensive repository metadata
- Fetches trending/popular repositories
- Handles rate limiting and retries
- Stores raw data in Cosmos DB for processing

**Usage:**
```bash
python pipelines/github_scraper.py
```

**Schedule:** Weekly (can be configured via cron or GitHub Actions)

---

### 2. **Cosmos to MongoDB ETL** (`cosmos_to_mongo_etl.py`)
Transforms raw repository data from Cosmos DB into cleaned MongoDB format.

**Features:**
- Data validation and cleaning
- Quality score calculation
- Transforms to structured schema
- Bulk upsert to MongoDB

**Usage:**
```bash
python pipelines/cosmos_to_mongo_etl.py
```

**Data Flow:** Cosmos DB (raw) → MongoDB (cleaned)

---

### 3. **Qdrant Embedding Service** (`qdrant_embedding_service.py`)
Generates OpenAI embeddings for repository data and stores in Qdrant.

**Features:**
- Generates text representations of repositories
- Creates embeddings using OpenAI API
- Associates vectors with MongoDB entries
- Stores in Qdrant for similarity search

**Usage:**
```bash
python pipelines/qdrant_embedding_service.py
```

**Dependencies:** OpenAI API key required

---

### 4. **Redis Chat Cache** (`redis_chat_cache.py`)
Manages caching layer for chatbot operations.

**Features:**
- Chat history storage
- User session management
- Query result caching
- User preference storage
- 1-week TTL on all cached data

**Usage:**
```python
from pipelines.redis_chat_cache import RedisChatCache

cache = RedisChatCache()
cache.save_chat_message(session_id, "user", "Hello!")
cache.save_user_preferences(user_id, {"language_filter": ["Python"]})
```

---

## Pipeline Execution Order

For initial setup or full data refresh:

```bash
# 1. Scrape GitHub data
python pipelines/github_scraper.py

# 2. Clean and transform to MongoDB
python pipelines/cosmos_to_mongo_etl.py

# 3. Generate embeddings
python pipelines/qdrant_embedding_service.py
```

## Automation

### GitHub Actions
The `.github/workflows/deploy-dev.yml` workflow automatically:
1. Deploys code to DEV server
2. Initializes all databases
3. Sets up environment variables
4. Creates Docker networks
5. Starts database containers

Triggered on push to `dev` branch.

### Cron Jobs (Optional)
Add to server crontab for scheduled execution:

```cron
# Run GitHub scraper weekly (Sundays at 2 AM)
0 2 * * 0 cd /path/to/git-query && python3 pipelines/github_scraper.py

# Run ETL pipeline daily (at 3 AM)
0 3 * * * cd /path/to/git-query && python3 pipelines/cosmos_to_mongo_etl.py

# Generate embeddings daily (at 4 AM)
0 4 * * * cd /path/to/git-query && python3 pipelines/qdrant_embedding_service.py
```

## Environment Variables

All pipelines require the following environment variables:

```bash
# Database connections
MONGODB_URL=mongodb://user:pass@localhost:27017/gitquery
COSMOS_DB_URL=https://localhost:8081
COSMOS_DB_KEY=your-cosmos-key
QDRANT_URL=http://localhost:6333
REDIS_URL=redis://localhost:6379

# API Keys
OPENAI_API_KEY=your-openai-key
GITHUB_TOKEN=your-github-token
```

## Monitoring

Check pipeline logs:
```bash
# View logs from last run
tail -f /var/log/git-query/pipelines.log

# Check Redis cache stats
python3 -c "from pipelines.redis_chat_cache import RedisChatCache; print(RedisChatCache().get_cache_stats())"
```

## Dependencies

Install required packages:
```bash
pip install -r requirements.txt
pip install -r DB/requirements.txt
```

Key dependencies:
- `openai` - Embedding generation
- `pymongo` - MongoDB operations
- `qdrant-client` - Vector storage
- `redis` - Cache layer
- `requests` - GitHub API

---

For more details, see individual pipeline files.
