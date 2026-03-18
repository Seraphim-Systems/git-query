# Recommender Service

Repository recommendation system with hybrid retrieval, cross-encoder reranking, personalization, and A/B testing.

## Architecture

```
Query -> RETRIEVAL -> RERANKING -> PERSONALIZATION -> Results
         (fast)       (precise)    (user prefs)

Retrieval:      bi-encoder embeddings (Qdrant) + keyword search (MongoDB)
                merged via Reciprocal Rank Fusion (RRF)

Reranking:      cross-encoder scores top candidates (pretrained for MVP,
                DS replaces with custom LightGBM ranker later)

Personalization: language/topic preference boost from interaction history
                 (rule-based, activates after 5+ user interactions)
```

### Engines

| Engine | Strategy | When to use |
|--------|----------|-------------|
| `baseline` | MongoDB keyword regex search | A/B test control group |
| `hybrid` | Semantic + keyword + RRF + cross-encoder rerank | Default production |
| `personalized` | Hybrid + user preference boost | Users with 5+ interactions |

## Project Structure

```
src/recommender/
  api.py                   # FastAPI app (10 endpoints)
  config.py                # RecommenderSettings (pydantic-settings)
  models.py                # Pydantic models
  database.py              # MongoDB + Qdrant + Redis clients
  engines/
    base.py                # Abstract engine interface
    baseline.py            # Keyword search only
    hybrid.py              # Semantic + keyword + RRF + reranking
    personalized.py        # Hybrid + user preference boost
  services/
    embedding_service.py   # Bi-encoder inference (all-MiniLM-L6-v2)
    reranker_service.py    # Cross-encoder scoring (ms-marco-MiniLM-L-6-v2)
    personalization_service.py  # Learn prefs from interactions
    ab_test_service.py     # Consistent hash-based variant assignment
    registry_service.py    # Model lifecycle (register/promote/archive)
  data/
    dataset.py             # RepoDataset: fetch, cache, DataFrame
    features.py            # FeatureExtractor: numeric features for ML
  training/
    unified_pipeline.py    # Indexing pipeline: fetch -> embed -> Qdrant
    utils.py               # Canonical prepare_repo_text()
    embedding_trainer.py   # Fine-tune bi-encoder (needs interaction data)
    reranker_trainer.py    # Fine-tune cross-encoder (needs interaction data)
    evaluator.py           # Precision@K, NDCG@K, MRR, CTR
  scripts/
    upload_embeddings.py   # Upload .npy vectors to Qdrant
    create_ab_test.py      # Create A/B test config in MongoDB
  notebooks/
    01_exploration_and_baseline.ipynb  # DS workflow example
```

## Quick Start

### Run the API

```bash
python -m src.recommender
# Starts on http://localhost:8095
```

### Run the Indexing Pipeline (Docker)

```bash
# Build
docker build -f infrastructure/docker/Dockerfile.training -t gitquery-training .

# Run (fetches repos, generates embeddings, uploads to Qdrant)
docker run --rm --env-file .env \
  -e MAX_REPOS=1000 \
  -e SKIP_IF_NO_NEW_DATA=false \
  -v $(pwd)/models:/app/models \
  gitquery-training
```

### Environment Variables

```bash
# Required for indexing pipeline
API_BASE_URL=https://base-url
APIKEY_MONGODB=your-api-key
APIKEY_QDRANT=your-api-key

# Optional
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
BATCH_SIZE=32
MAX_REPOS=                # empty = all
SKIP_IF_NO_NEW_DATA=true
SKIP_QDRANT_UPLOAD=false
QDRANT_COLLECTION=repositories_embeddings
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/recommend` | Get recommendations (routes via A/B test) |
| POST | `/recommend/explain/{repo_id}` | Why was this recommended? |
| POST | `/interaction` | Log click/save/dismiss (updates prefs in background) |
| GET | `/preferences/{user_id}` | User preference profile |
| GET | `/metrics/{variant}` | Evaluation metrics for variant |
| GET | `/ab-test` | Active A/B test config |
| POST | `/admin/cache/clear` | Clear Redis cache |
| GET | `/admin/engines` | List engines |
| GET | `/admin/models` | List registered models |
| POST | `/admin/models/reload` | Hot-reload active models |
| POST | `/admin/models/promote/{model_id}` | Promote candidate to active |
| GET | `/health` | Health check |

## What's Working vs. Planned

### Working Now (MVP)
- Baseline engine (keyword search)
- Indexing pipeline: fetch repos -> pretrained embeddings -> Qdrant upload
- All API endpoints
- Interaction logging + preference learning
- A/B test framework
- Model registry + promotion
- Redis caching
- Cross-encoder reranking (pretrained)

### Needs User Data (Phase 2)
- Personalized engine (needs 5+ interactions per user)
- Embedding fine-tuning (embedding_trainer.py)
- Cross-encoder fine-tuning (reranker_trainer.py)
- Offline evaluation with real ground truth

### Data Science Builds (Phase 3)
- Custom ranking model (see DS section below)

---

## Data Science Guide

### Your Goal

Build a **ranking model** that replaces the pretrained cross-encoder reranker with a custom model trained on repo features.

### The Pipeline

```
Step 1 (Retrieval - ML Eng owns):
  bi-encoder + Qdrant + MongoDB keyword -> ~100 candidates

Step 2 (Feature Extraction - you build):
  For each candidate: semantic_score, keyword_score, stars, recency,
  topic overlap, readme quality, fork ratio, etc.

Step 3 (Ranking Model - you build):
  LightGBM/XGBoost ranker: features -> relevance score -> top 10

Step 4 (Personalization - ML Eng owns):
  User preference boost (rule-based)
```

### Getting Started

```python
# In a Jupyter notebook
import os
from dotenv import load_dotenv
load_dotenv()

from src.recommender.data import RepoDataset, FeatureExtractor

# Load data
ds = RepoDataset.from_gateway(
    url=os.getenv("API_BASE_URL"),
    api_key=os.getenv("GATEWAY_API_KEY"),
    max_repos=5000
)
ds.save("src/recommender/notebooks/data/repos.parquet")

# Explore
df = ds.to_dataframe()
ds.summary()

# Extract features
fe = FeatureExtractor()
features = fe.extract_all(df, query="python web framework")

# Train baseline model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)
model = GradientBoostingRegressor(n_estimators=100)
model.fit(X_train, y_train)
```

See `notebooks/01_exploration_and_baseline.ipynb` for the full workflow.

### What Model to Build

**LightGBM ranker** with `lambdarank` objective. Not a neural network.

Features to experiment with:
- `semantic_score` - bi-encoder similarity (from EmbeddingService)
- `cross_encoder_score` - cross-encoder similarity (from RerankerService)
- `stars_log`, `forks_log` - popularity signals
- `days_since_update` - freshness
- `topic_overlap` - query-repo topic match
- `readme_length` - documentation quality
- `fork_star_ratio` - engagement signal
- `has_license`, `is_permissive_license` - quality signals
- `language_encoded` - one-hot top languages

### Labels (Cold Start)

No users yet, so create synthetic labels:
- **Option A**: Topic overlap as relevance proxy (query terms in repo topics)
- **Option B**: Manual labeling of 50-100 test queries
- **Option C**: Star count as quality signal, combined with topic match

### Evaluation

Use `src/recommender/training/evaluator.py`:
- Precision@K, Recall@K (K = 1, 5, 10, 20)
- NDCG@K (ranking quality)
- MRR (mean reciprocal rank)

### Deploying Your Model

1. Train and export model (`.pkl` or `.txt`)
2. Register via `POST /admin/models` or `ModelRegistryService.register_model()`
3. A/B test against baseline: `scripts/create_ab_test.py`
4. Promote winner: `POST /admin/models/promote/{model_id}`
5. Hot-reload: `POST /admin/models/reload`

---

## Adding a New Engine

```python
from src.recommender.engines.base import RecommendationEngine

class MyEngine(RecommendationEngine):
    def __init__(self):
        super().__init__(name="my_engine", version="1.0.0")

    async def recommend(self, request):
        # Your ranking logic
        pass

    async def explain(self, repo_id, request):
        return {"engine": self.name, "method": "custom"}
```

Register in `api.py` lifespan, then use via A/B test variant assignment.
