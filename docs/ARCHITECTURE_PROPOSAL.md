# Git-Query: Dockerized Architecture Proposal

## Executive Summary

This proposal outlines a comprehensive, production-ready architecture for git-query featuring:
- **6 containerized services** with clear separation of concerns
- **Automated ML pipeline** for recommendation model training
- **Event-driven architecture** for real-time data processing
- **Scalable vector search** using Qdrant for recommendations

---

## 1. System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         NGINX (Web Container)                    │
│  ┌──────────────┐    ┌────────────────────────────────────┐   │
│  │   Frontend   │    │  Reverse Proxy & Load Balancer     │   │
│  │  (React/Vue) │    │  - Routes /api → MCP Server        │   │
│  └──────────────┘    │  - Routes /recommend → Reco API     │   │
│                      │  - Serves static files               │   │
│                      └────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                               ↓
        ┌──────────────────────┼──────────────────────┐
        ↓                      ↓                       ↓
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ MCP Server   │      │   Reco API   │      │  Data Worker │
│ (FastAPI)    │      │  (FastAPI)   │      │  (Celery)    │
│              │      │              │      │              │
│ - Tools API  │      │ - Recommend  │      │ - Scraping   │
│ - Chatbot    │      │ - Search     │      │ - ETL        │
│ - Admin      │      │ - Analytics  │      │ - Training   │
└──────┬───────┘      └──────┬───────┘      └──────┬───────┘
       │                     │                      │
       └─────────────────────┼──────────────────────┘
                             ↓
        ┌────────────────────┼────────────────────┐
        ↓                    ↓                     ↓
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   MongoDB    │    │   Qdrant     │    │    Redis     │
│              │    │              │    │              │
│ - Raw Data   │    │ - Embeddings │    │ - Cache      │
│ - Training   │    │ - Vector DB  │    │ - Queue      │
│ - Logs       │    │ - Search     │    │ - Sessions   │
└──────────────┘    └──────────────┘    └──────────────┘
```

---

## 2. Container Services Breakdown

### 2.1 **Web Container** (nginx + frontend)
**Location:** `infrastructure/docker/Dockerfile.web`

**Responsibilities:**
- Serve static frontend files (React/Vue SPA)
- Reverse proxy to backend services
- SSL termination (production)
- Rate limiting & security headers
- Health checks

**Nginx Configuration Updates:**
```nginx
location /api/ {
    proxy_pass http://mcp-server:8000/;
}

location /recommend/ {
    proxy_pass http://recommender-api:8001/;
}

location /admin/ {
    proxy_pass http://mcp-server:8000/admin/;
}

location /ws {
    proxy_pass http://mcp-server:8000/ws;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
}
```

### 2.2 **MCP Server Container** (existing)
**Location:** `src/mcp/server.py`

**Current Logic (keep):**
- Tool execution API
- Chatbot client interface
- Admin endpoints

**Add New Logic:**
```python
src/mcp/
├── server.py              # Main FastAPI app (existing)
├── routers/
│   ├── tools.py          # Tool execution endpoints
│   ├── chatbot.py        # Chat completions
│   └── admin.py          # Admin operations
├── middleware/
│   ├── auth.py           # JWT/API key validation
│   └── rate_limit.py     # Request throttling
└── services/
    ├── tool_executor.py  # Tool orchestration
    └── session_manager.py # User session handling
```

### 2.3 **Recommender API Container** (NEW)
**Location:** `src/recommender/api/server.py`

**Purpose:** Dedicated recommendation inference service

**Application Logic:**
```python
src/recommender/
├── api/
│   ├── server.py         # FastAPI recommendation server
│   ├── routers/
│   │   ├── recommend.py  # GET /recommend
│   │   ├── search.py     # POST /search (vector search)
│   │   └── feedback.py   # POST /feedback (user actions)
│   └── models/
│       └── schemas.py    # Request/response models
├── engine/
│   ├── inference.py      # Model inference logic
│   ├── vector_store.py   # Qdrant client wrapper
│   ├── ranking.py        # Re-ranking algorithms
│   └── filters.py        # Business rule filters
└── models/
    └── embedder.py       # Text → vector embedding
```

**Key Endpoints:**
```python
# GET /recommend?user_id={id}&limit=10&context={git_repo}
# POST /search { "query": "python ML repos", "filters": {...} }
# POST /feedback { "user_id": "...", "item_id": "...", "action": "click" }
```

### 2.4 **Data Worker Container** (NEW)
**Location:** `src/processing/worker.py`

**Purpose:** Async task processing using Celery

**Application Logic:**
```python
src/processing/
├── worker.py             # Celery worker entry point
├── tasks/
│   ├── scraping.py       # GitHub repo scraping
│   ├── embedding.py      # Generate embeddings
│   ├── indexing.py       # Insert into Qdrant
│   └── cleanup.py        # Data maintenance
├── pipelines/
│   ├── ingestion.py      # Raw data → MongoDB
│   ├── transformation.py # Data cleaning/enrichment
│   └── vectorization.py  # Text → embeddings
└── schedulers/
    ├── cron_jobs.py      # Periodic tasks (scraping, retraining)
    └── event_handlers.py # React to DB changes
```

**Celery Tasks:**
```python
@celery.task
def scrape_github_repos(search_query: str):
    """Scrape GitHub repos and store in MongoDB"""
    
@celery.task
def generate_embeddings(doc_ids: list[str]):
    """Generate embeddings for new documents"""
    
@celery.task
def update_qdrant_index(embeddings: list):
    """Bulk insert embeddings into Qdrant"""
    
@celery.task(bind=True)
def retrain_model(self, config: dict):
    """Trigger model retraining pipeline"""
```

### 2.5 **Training Service Container** (NEW)
**Location:** `src/recommender/training/trainer.py`

**Purpose:** Model training orchestration (triggered manually or on schedule)

**Application Logic:**
```python
src/recommender/training/
├── trainer.py            # Main training orchestrator
├── data_loader.py        # Load data from MongoDB
├── feature_engineering.py# Create training features
├── models/
│   ├── collaborative_filtering.py  # User-item interactions
│   ├── content_based.py            # Repo metadata/embeddings
│   └── hybrid.py                   # Combined model
├── evaluation.py         # Metrics & validation
└── export.py             # Save model artifacts
```

**Training Pipeline:**
```python
class ModelTrainer:
    def run_training_pipeline(self):
        # 1. Extract data from MongoDB
        data = self.extract_training_data()
        
        # 2. Feature engineering
        features = self.create_features(data)
        
        # 3. Train model
        model = self.train_model(features)
        
        # 4. Evaluate
        metrics = self.evaluate_model(model)
        
        # 5. Export to model registry
        if metrics['auc'] > 0.85:
            self.save_model(model)
            self.update_qdrant_index(model.embeddings)
```

---

## 3. Data Flow Architecture

### 3.1 Real-Time Recommendation Flow
```
User Request → Nginx → Recommender API
                           ↓
                    Fetch user profile (Redis cache)
                           ↓
                    Query Qdrant (vector similarity)
                           ↓
                    Re-rank & filter results
                           ↓
                    Return recommendations
```

### 3.2 Data Ingestion Pipeline
```
Scheduler (cron) → Data Worker → Scrape GitHub
                                      ↓
                              Store in MongoDB (raw)
                                      ↓
                              Process & Clean Data
                                      ↓
                              Generate Embeddings
                                      ↓
                              Index in Qdrant
```

### 3.3 Model Training Pipeline
```
Trigger (manual/scheduled) → Training Service
                                    ↓
                            Load data from MongoDB
                                    ↓
                            Feature engineering
                                    ↓
                            Train model
                                    ↓
                            Evaluate metrics
                                    ↓
                    Pass → Export model & Update Qdrant
                    Fail → Log & alert
```

---

## 4. Data Processing Pipelines

### 4.1 Scraping Pipeline
**Trigger:** Cron job (daily at 2 AM)
**Location:** `src/processing/tasks/scraping.py`

```python
from celery import chain

# Chain of tasks
scraping_pipeline = chain(
    scrape_github_repos.s(query="machine learning"),
    clean_repo_data.s(),
    extract_metadata.s(),
    store_in_mongodb.s()
)
scraping_pipeline.apply_async()
```

### 4.2 Embedding Generation Pipeline
**Trigger:** Event-driven (new MongoDB documents)
**Location:** `src/processing/pipelines/vectorization.py`

```python
# MongoDB Change Stream → Celery Task
async def watch_new_documents():
    async for change in mongodb.repos.watch():
        if change['operationType'] == 'insert':
            generate_embeddings.delay(change['documentKey']['_id'])
```

### 4.3 Training Pipeline
**Trigger:** 
- Manual via API: `POST /admin/trigger-training`
- Scheduled: Weekly on Sunday
**Location:** `src/recommender/training/trainer.py`

```python
# Celery beat schedule
from celery.schedules import crontab

app.conf.beat_schedule = {
    'retrain-weekly': {
        'task': 'recommender.training.retrain_model',
        'schedule': crontab(day_of_week=0, hour=3),
    },
}
```

---

## 5. Database Usage Strategy

### MongoDB (Primary Data Store)
**Collections:**
```python
- repos                # Scraped repository data
  ├── repo_id
  ├── name, description, stars, language
  ├── scraped_at
  └── metadata
  
- user_interactions    # Clicks, views, clones
  ├── user_id
  ├── repo_id
  ├── action_type
  └── timestamp
  
- training_datasets    # Historical training data
  └── version, features, labels, created_at
  
- model_artifacts      # Model metadata & configs
  └── version, metrics, hyperparameters, path
```

### Qdrant (Vector Search)
**Collections:**
```python
- repository_embeddings  # Repo description embeddings
  └── vector[768], payload: {repo_id, metadata}
  
- code_embeddings        # Code snippet embeddings
  └── vector[768], payload: {repo_id, file_path}
  
- user_preference_embeddings  # User interest vectors
  └── vector[768], payload: {user_id, preferences}
```

### Redis (Cache & Queue)
**Usage:**
- Cache: User profiles, recent recommendations
- Queue: Celery task broker
- Rate limiting: API throttling
- Session storage: User sessions

---

## 6. Automated Server Operations

### 6.1 Celery Beat Scheduler
**Location:** `src/processing/schedulers/cron_jobs.py`

```python
from celery import Celery
from celery.schedules import crontab

app = Celery('git-query')

# Scheduled tasks
app.conf.beat_schedule = {
    # Daily scraping
    'scrape-trending-repos': {
        'task': 'tasks.scraping.scrape_github_repos',
        'schedule': crontab(hour=2, minute=0),
        'args': ('trending', 100)
    },
    
    # Weekly model retraining
    'retrain-recommender': {
        'task': 'recommender.training.retrain_model',
        'schedule': crontab(day_of_week=0, hour=3),
    },
    
    # Hourly embedding generation
    'generate-pending-embeddings': {
        'task': 'tasks.embedding.process_pending',
        'schedule': crontab(minute=0),  # Every hour
    },
    
    # Daily cleanup
    'cleanup-old-data': {
        'task': 'tasks.cleanup.remove_old_entries',
        'schedule': crontab(hour=4, minute=0),
        'kwargs': {'days': 180}
    },
}
```

### 6.2 Event-Driven Automation
**MongoDB Change Streams:**
```python
# Auto-trigger embedding generation
async def auto_embed_new_repos():
    async with mongodb.repos.watch() as stream:
        async for change in stream:
            if change['operationType'] == 'insert':
                generate_embeddings.delay(change['fullDocument']['_id'])
```

### 6.3 Model Retraining Triggers
```python
# Trigger conditions
def should_retrain() -> bool:
    # Check metrics
    recent_ctr = calculate_ctr(days=7)
    if recent_ctr < 0.05:  # CTR dropped
        return True
    
    # Check data freshness
    last_training = get_last_training_date()
    if (datetime.now() - last_training).days > 7:
        return True
    
    # Check new data volume
    new_repos_count = mongodb.repos.count_documents({
        'created_at': {'$gte': last_training}
    })
    if new_repos_count > 1000:  # Significant new data
        return True
    
    return False
```

---

## 7. Updated Docker Compose Structure

### Add New Services:
```yaml
services:
  # ... existing services ...
  
  # Recommendation API
  recommender-api:
    build:
      context: .
      dockerfile: infrastructure/docker/Dockerfile.recommender
    container_name: git-query-recommender-api
    expose:
      - "8001"
    environment:
      - API_HOST=0.0.0.0
      - API_PORT=8001
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379
      - MODEL_PATH=/models
    volumes:
      - model-artifacts:/models:ro
    depends_on:
      - qdrant
      - redis
    networks:
      - git-query-app-network
      - git-query-db-network
    restart: unless-stopped

  # Data Worker (Celery)
  data-worker:
    build:
      context: .
      dockerfile: infrastructure/docker/Dockerfile.worker
    container_name: git-query-data-worker
    command: celery -A src.processing.worker worker --loglevel=info
    environment:
      - CELERY_BROKER_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
      - CELERY_RESULT_BACKEND=redis://:${REDIS_PASSWORD}@redis:6379/1
      - MONGODB_URL=mongodb://${MONGO_USER}:${MONGO_PASSWORD}@mongodb:27017/gitquery
      - QDRANT_HOST=qdrant
      - GITHUB_TOKEN=${GITHUB_TOKEN}
    depends_on:
      - mongodb
      - redis
      - qdrant
    networks:
      - git-query-db-network
    restart: unless-stopped

  # Celery Beat (Scheduler)
  celery-beat:
    build:
      context: .
      dockerfile: infrastructure/docker/Dockerfile.worker
    container_name: git-query-celery-beat
    command: celery -A src.processing.worker beat --loglevel=info
    environment:
      - CELERY_BROKER_URL=redis://:${REDIS_PASSWORD}@redis:6379/0
    depends_on:
      - redis
      - data-worker
    networks:
      - git-query-db-network
    restart: unless-stopped

  # Training Service (On-demand)
  training-service:
    build:
      context: .
      dockerfile: infrastructure/docker/Dockerfile.training
    container_name: git-query-training
    environment:
      - MONGODB_URL=mongodb://${MONGO_USER}:${MONGO_PASSWORD}@mongodb:27017/gitquery
      - QDRANT_HOST=qdrant
      - MODEL_OUTPUT_PATH=/models
    volumes:
      - model-artifacts:/models
    depends_on:
      - mongodb
      - qdrant
    networks:
      - git-query-db-network
    profiles:
      - training  # Only start when explicitly requested

volumes:
  model-artifacts:
    name: git-query-model-artifacts
```

---

## 8. Deployment & Operations

### 8.1 Starting Services
```bash
# Start infrastructure (databases, db-api, nginx)
docker compose -f infrastructure/docker/docker-compose.base.yml -f infrastructure/docker/docker-compose.db.yml up -d

# Start pipelines
docker compose -f infrastructure/docker/docker-compose.base.yml -f infrastructure/docker/docker-compose.pipelines.yml up -d

# Start MCP servers
docker compose -f infrastructure/docker/docker-compose.base.yml -f infrastructure/docker/docker-compose.mcp.yml up -d

# Start recommendation engine
docker compose -f infrastructure/docker/docker-compose.base.yml -f infrastructure/docker/docker-compose.reco.yml up -d

# Trigger training manually
docker compose -f infrastructure/docker/docker-compose.base.yml -f infrastructure/docker/docker-compose.training.yml up -d
```

### 8.2 Monitoring
```python
# Add monitoring endpoints
GET /health          # Service health
GET /metrics         # Prometheus metrics
GET /admin/tasks     # View Celery tasks
GET /admin/models    # Model versions & metrics
```

### 8.3 CI/CD Integration
```yaml
# .github/workflows/deploy.yml
- name: Build and deploy
  run: |
    docker compose build
    docker compose up -d
    docker compose run --rm training-service  # Initial training
```

---

## 9. Implementation Priority

### Phase 1: Foundation (Week 1-2)
1. ✅ Set up nginx routing configuration
2. ✅ Implement recommender API skeleton
3. ✅ Create Celery worker infrastructure
4. ✅ Design MongoDB collections

### Phase 2: Data Pipeline (Week 3-4)
1. Implement scraping tasks
2. Build embedding generation pipeline
3. Set up MongoDB → Qdrant sync
4. Configure Celery Beat schedules

### Phase 3: ML Pipeline (Week 5-6)
1. Implement training service
2. Build evaluation metrics
3. Create model registry
4. Set up automated retraining

### Phase 4: Integration (Week 7-8)
1. Connect frontend to recommender API
2. Implement feedback loop
3. Add monitoring & logging
4. Performance optimization

---

## 10. Key Benefits

✅ **Separation of Concerns:** Each container has a single responsibility
✅ **Scalability:** Scale recommendation API independently from data processing
✅ **Automation:** Fully automated data ingestion and model training
✅ **Resilience:** Failed tasks can retry; services restart automatically
✅ **Maintainability:** Clear code organization and deployment strategy
✅ **Performance:** Redis caching + Qdrant vector search = fast recommendations

---

## 11. Next Steps

1. **Review & Approve Architecture:** Discuss trade-offs and alternatives
2. **Create Dockerfiles:** Build new container images
3. **Implement Core Services:** Start with recommender API + data worker
4. **Deploy Incrementally:** Roll out services one at a time
5. **Monitor & Iterate:** Collect metrics and optimize

---

**Questions to Address:**
- What recommendation algorithm (collaborative filtering, content-based, hybrid)?
- What ML framework (TensorFlow, PyTorch, scikit-learn)?
- What embedding model (Sentence-BERT, OpenAI, custom)?
- What's the expected request volume (affects scaling strategy)?
