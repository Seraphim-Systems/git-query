# Recommendation System

AI-powered repository recommendation system with hybrid retrieval, personalization, and A/B testing capabilities.

## Overview

This recommendation system implements the Netflix-style approach for GitHub repositories:
- **Hybrid Retrieval**: Combines semantic embeddings (meaning) with keyword search (exact matches)
- **Cross-Encoder Reranking**: Accurately ranks top candidates
- **Personalization**: Learns user preferences from interactions (language, topics)
- **A/B Testing**: Built-in support for testing different recommendation variants
- **Constraint Enforcement**: User filters (language, stars, license) are never violated

## Architecture

The system follows SOLID principles for easy extension and A/B testing:

```
┌─────────────────────────────────────────────────┐
│          Recommendation Engines                 │
│  (Easy to add new variants for A/B testing)     │
├─────────────────────────────────────────────────┤
│  BaselineEngine    │  Keyword search only       │
│  HybridEngine      │  Embeddings + Keywords     │
│  PersonalizedEngine│  Hybrid + User Preferences │
└─────────────────────────────────────────────────┘
           ↓                    ↓
┌──────────────────┐   ┌────────────────┐
│  Embedding       │   │  Reranker      │
│  Service         │   │  Service       │
│  (Bi-encoder)    │   │  (Cross-enc)   │
└──────────────────┘   └────────────────┘
```

## Key Features

### 1. Hybrid Retrieval
- **Semantic Search**: Understands "Caesar Cipher" ≈ "Shift Cipher"
- **Keyword Search**: Catches exact term matches
- **RRF Fusion**: Reciprocal Rank Fusion combines both approaches

### 2. Personalization
- Learns from user interactions (clicks, saves, thumbs up/down)
- Boosts repos in preferred languages when language not specified
- Only applies with minimum interaction threshold
- Never overrides user constraints

### 3. A/B Testing
- Consistent hash-based user assignment
- Multiple variants can run simultaneously
- Traffic splitting configuration
- Automatic metrics collection per variant

### 4. Training & Evaluation
- Offline evaluation with Precision@K, Recall@K, NDCG@K, MRR
- Shadow mode testing before deployment
- Continuous learning from user feedback
- Model versioning and rollback support

## Setup

### Prerequisites
- Python 3.11+
- MongoDB (for data storage)
- Qdrant (for vector search)
- Redis (for caching)

### Environment Variables

Create a `.env` file (see `.env.example`):

```bash
# Server
RECOMMENDER_HOST=0.0.0.0
RECOMMENDER_PORT=8095
LOG_LEVEL=INFO

# Databases
MONGODB_URL=mongodb://admin:mongopass@localhost:27017/gitquery?authSource=admin
REDIS_URL=redis://:redispass@localhost:6379
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_API_KEY=

# Models
MODEL_PATH=./models
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
CROSS_ENCODER_MODEL_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2

# Retrieval Settings
HYBRID_SEARCH_TOP_K=100
RERANK_TOP_K=20
FINAL_TOP_K=10

# Personalization
ENABLE_PERSONALIZATION=true
PERSONALIZATION_WEIGHT=0.15
MIN_INTERACTIONS_FOR_PERSONALIZATION=5

# Caching
ENABLE_CACHE=true
CACHE_TTL_SECONDS=3600

# A/B Testing
AB_TEST_ENABLED=true
DEFAULT_VARIANT=baseline
```

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
python -m recommender
```

### Docker Deployment

```bash
# Build and run with docker-compose
cd infrastructure/docker
docker-compose -f docker-compose.base.yml -f docker-compose.reco.yml up
```

## API Endpoints

### Get Recommendations
```http
POST /recommend
Content-Type: application/json

{
  "query": "python web framework",
  "user_id": "user123",
  "language": "Python",
  "min_stars": 100,
  "top_k": 10,
  "enable_personalization": true
}
```

### Log User Interaction
```http
POST /interaction
Content-Type: application/json

{
  "user_id": "user123",
  "query": "python web framework",
  "repo_id": "repo456",
  "interaction_type": "click",
  "position_in_results": 3,
  "variant": "personalized"
}
```

### Get User Preferences
```http
GET /preferences/{user_id}
```

### Get Metrics
```http
GET /metrics/{variant}
```

### Health Check
```http
GET /health
```

## Metrics We Track

Based on your proposal, the system tracks:

1. **User Queries**: What users search for
2. **User Preferences**: Language preferences, topic interests
3. **Recommendations Shown**: What the system recommended
4. **User Choices**: What users actually clicked/saved
5. **Click-Through Rate**: % of recommendations clicked
6. **Feedback**: Thumbs up/down ratings
7. **Precision@K**: Are top K results relevant?
8. **NDCG@K**: Quality of ranking
9. **MRR**: How quickly users find what they want

## Training Pipeline

The training pipeline can be run on a schedule:

```python
from recommender.training import TrainingPipeline

pipeline = TrainingPipeline(variant="v2")
await pipeline.run_full_pipeline(
    train_embeddings=True,
    train_reranker=True,
    min_interactions=1000
)
```

Steps:
1. Extract training data from user interactions
2. Train embedding model on (query, positive_repo) pairs
3. Train cross-encoder on (query, repo, label) tuples
4. Evaluate in shadow mode
5. Deploy if performance improves

## Adding New Recommendation Engines

Thanks to SOLID principles, adding a new engine is simple:

```python
from recommender.engines.base import RecommendationEngine
from recommender.models import RecommendationRequest, RepositoryResult

class MyNewEngine(RecommendationEngine):
    def __init__(self):
        super().__init__(name="my_new_engine", version="1.0.0")
    
    async def recommend(self, request: RecommendationRequest) -> List[RepositoryResult]:
        # Your recommendation logic here
        pass
    
    async def explain(self, repo_id: str, request: RecommendationRequest) -> Dict:
        # Explanation logic
        pass
```

Then register it in `api.py` and use it in A/B tests!

## Project Structure

```
recommender/
├── __init__.py
├── __main__.py          # Entry point
├── api.py               # FastAPI application
├── config.py            # Settings
├── models.py            # Pydantic models
├── database.py          # Database clients
├── engines/             # Recommendation engines (SOLID)
│   ├── base.py         # Abstract base class
│   ├── baseline.py     # Keyword search
│   ├── hybrid.py       # Embeddings + Keywords
│   └── personalized.py # Hybrid + Personalization
├── services/            # Business logic
│   ├── embedding_service.py
│   ├── reranker_service.py
│   ├── personalization_service.py
│   └── ab_test_service.py
└── training/            # Training pipelines
    ├── pipeline.py
    ├── embedding_trainer.py
    ├── reranker_trainer.py
    └── evaluator.py
```

## Development

### Running Tests
```bash
# TODO: Add tests
pytest tests/
```

### Running Locally
```bash
# Start dependencies
docker-compose -f infrastructure/docker/docker-compose.db.yml up

# Run service
python -m recommender
```

## Future Extensions

Based on your proposal, potential additions:
- [ ] Implement data collection for all key metrics
- [ ] Add online learning for continuous improvement
- [ ] Implement diversity in recommendations
- [ ] Add explainability features
- [ ] Multi-armed bandit for dynamic A/B testing
- [ ] Knowledge graph integration for better semantic understanding

## License

See main project LICENSE

