# Recommender System - Complete Setup Summary

## 🎯 What Was Built

A production-ready, Netflix-style recommendation system for GitHub repositories with:

### ✅ Core Features Implemented

1. **Hybrid Retrieval System**
   - Semantic search using embeddings (understands "Caesar Cipher" ≈ "Shift Cipher")
   - Keyword search for exact matches
   - Reciprocal Rank Fusion (RRF) to combine both approaches
   - Cross-encoder reranking for accurate top-K results

2. **Personalization Engine**
   - Learns from user interactions (clicks, saves, thumbs up/down)
   - Boosts repos in user's preferred languages
   - Only applies with minimum interaction threshold (default: 5)
   - **Never violates user constraints** (language, stars, license filters)

3. **A/B Testing Framework**
   - Consistent hash-based user assignment
   - Multiple variants can run simultaneously
   - Traffic splitting configuration
   - Automatic metrics collection per variant

4. **Training & Evaluation Pipeline**
   - Offline evaluation metrics (Precision@K, Recall@K, NDCG@K, MRR)
   - Shadow mode testing before deployment
   - Model versioning and rollback support
   - Continuous learning from user feedback

5. **SOLID Architecture**
   - Easy to add new recommendation engines
   - Each engine follows the same interface
   - Perfect for rapid A/B testing iterations

## 📁 Project Structure

```
src/recommender/
├── api.py                          # FastAPI application with all endpoints
├── config.py                       # Configuration settings
├── models.py                       # Pydantic models
├── database.py                     # Database clients (MongoDB, Qdrant, Redis)
├── __init__.py                     # Package initialization
├── __main__.py                     # Entry point
│
├── engines/                        # Recommendation engines (SOLID)
│   ├── __init__.py
│   ├── base.py                    # Abstract base class
│   ├── baseline.py                # Keyword search only
│   ├── hybrid.py                  # Embeddings + Keywords + RRF
│   └── personalized.py            # Hybrid + User preferences
│
├── services/                       # Business logic services
│   ├── __init__.py
│   ├── embedding_service.py       # Semantic embeddings (bi-encoder)
│   ├── reranker_service.py        # Cross-encoder reranking
│   ├── personalization_service.py # User preference learning
│   └── ab_test_service.py         # A/B test management
│
├── training/                       # Training pipelines
│   ├── __init__.py
│   ├── pipeline.py                # Main training orchestrator
│   ├── embedding_trainer.py       # Train embedding models
│   ├── reranker_trainer.py        # Train cross-encoders
│   └── evaluator.py               # Evaluation metrics
│
├── scripts/                        # Utility scripts
│   ├── train.py                   # Run training pipeline
│   ├── evaluate.py                # Evaluate variants
│   ├── create_ab_test.py          # Create new A/B tests
│   └── quickstart.py              # Quick setup test
│
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
└── README.md                       # Complete documentation
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
cd src/recommender
cp .env.example .env
# Edit .env with your database URLs
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the Service

```bash
# Make sure MongoDB, Qdrant, and Redis are running
python -m recommender
```

The service will start on `http://localhost:8095`

### 4. Test the Setup

```bash
# Run quick start test
python scripts/quickstart.py

# Check health
curl http://localhost:8095/health
```

## 🔌 API Endpoints

### Get Recommendations
```http
POST /recommend
{
  "query": "python web framework",
  "user_id": "user123",
  "language": "Python",
  "min_stars": 100,
  "top_k": 10
}
```

### Log Interaction
```http
POST /interaction
{
  "user_id": "user123",
  "query": "python web framework",
  "repo_id": "repo456",
  "interaction_type": "click",
  "position_in_results": 3
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

## 📊 Metrics Tracked (As Per Your Proposal)

All the metrics you specified are implemented:

1. ✅ **User Queries** - Stored with each interaction
2. ✅ **User Preferences** - Language and topic preferences learned automatically
3. ✅ **Recommendations Shown** - All recommendations are logged
4. ✅ **User Choices** - Click, save, dismiss, thumbs up/down tracked
5. ✅ **Click-Through Rate** - Calculated in evaluation metrics
6. ✅ **Feedback** - Thumbs up/down with weights (positive vs negative signals)
7. ✅ **Precision@K** - Computed for K=1,5,10,20
8. ✅ **Recall@K** - Computed for K=1,5,10,20
9. ✅ **NDCG@K** - Ranking quality metric
10. ✅ **MRR** - Mean Reciprocal Rank

## 🔬 How It Works (Netflix Analogy)

Just like Netflix recommends movies based on your viewing history and search:

1. **User searches "python web framework"**
   - Hybrid search finds repos by semantic meaning AND exact keywords
   - Gets top 100 candidates

2. **Apply Hard Filters**
   - If user specified `language=Python`, only Python repos pass
   - If `min_stars=100`, only repos with 100+ stars pass
   - **Filters are NEVER violated**

3. **Rerank Top Candidates**
   - Cross-encoder scores each repo against the query
   - More accurate but slower, so only for top 20

4. **Personalization (if enabled)**
   - If user frequently clicks C++ repos, boost C++ repos
   - Only applies if user has 5+ interactions
   - Never overrides filters

5. **Return Top K Results**
   - Default: 10 results
   - Each with relevance score and explanation

## 🧪 A/B Testing

Create a new A/B test:

```bash
python scripts/create_ab_test.py \
  --name "Baseline vs Hybrid" \
  --description "Test hybrid retrieval" \
  --variants baseline hybrid \
  --splits 0.5 0.5 \
  --duration-days 14
```

Users are consistently assigned to variants using hash-based assignment.

## 🏋️ Training Pipeline

Run the training pipeline:

```bash
python scripts/train.py \
  --variant v2 \
  --train-embeddings \
  --train-reranker \
  --min-interactions 1000
```

The pipeline:
1. Extracts training data from user interactions
2. Creates (query, positive_repo, negative_repo) pairs
3. Trains/fine-tunes embedding model
4. Trains cross-encoder on relevance pairs
5. Evaluates in shadow mode
6. Deploys if performance improves

## 📈 Adding New Engines (SOLID)

Thanks to SOLID principles, adding a new recommendation variant is easy:

```python
from recommender.engines.base import RecommendationEngine

class MyAwesomeEngine(RecommendationEngine):
    def __init__(self):
        super().__init__(name="awesome", version="1.0.0")
    
    async def recommend(self, request):
        # Your logic here
        pass
    
    async def explain(self, repo_id, request):
        # Explanation logic
        pass
```

Register in `api.py` and use in A/B tests immediately!

## 🐳 Docker Support

The service is already dockerized:

- **Dockerfile**: `infrastructure/docker/Dockerfile.recommender`
- **Compose**: `infrastructure/docker/docker-compose.reco.yml`

Start with:
```bash
docker-compose -f docker-compose.base.yml -f docker-compose.reco.yml up
```

## 🔄 Integration Points

The recommender integrates with:

1. **MongoDB** - Stores repos, interactions, preferences
2. **Qdrant** - Vector search for semantic embeddings
3. **Redis** - Caching and session management
4. **MCP Server** (optional) - Can be called from chatbot
5. **Gateway** - Expose via API gateway

## 📝 Next Steps

1. **Populate Data**
   - Add repositories to MongoDB
   - Generate embeddings and store in Qdrant

2. **Start Collecting Interactions**
   - Integrate with frontend to log clicks, saves, etc.

3. **Train Initial Models**
   - Once you have ~1000 interactions, run training pipeline

4. **Run A/B Tests**
   - Compare baseline vs hybrid vs personalized

5. **Monitor & Iterate**
   - Check metrics, improve models, add new variants

## 🎓 Key Design Decisions

### Why This Architecture?

1. **SOLID Principles** - Easy to extend without breaking existing code
2. **Async/Await** - Non-blocking I/O for better performance
3. **Pydantic Models** - Type safety and validation
4. **Modular Services** - Each service has one responsibility
5. **A/B Testing First** - Built-in experimentation framework

### Why These Models?

- **sentence-transformers/all-MiniLM-L6-v2** - Fast, small, good enough for MVP
- **cross-encoder/ms-marco-MiniLM-L-6-v2** - Proven reranker for search tasks
- Can be swapped for larger/better models as needed

## ⚠️ Important Notes

1. **Filters are Sacred** - Personalization NEVER overrides user constraints
2. **Cold Start** - New users get baseline recommendations until 5+ interactions
3. **Privacy** - User preferences are computed from behavior, not personal data
4. **Performance** - Caching enabled by default (1 hour TTL)
5. **Versioning** - All models are versioned for easy rollback

## 📚 Files You Should Read

1. **README.md** - Full API documentation and usage
2. **.env.example** - All configuration options explained
3. **engines/base.py** - Understanding the engine interface
4. **api.py** - All available endpoints

## 🎉 What You Have Now

A complete, production-ready recommendation system that:

- ✅ Implements all your ML engineering requirements
- ✅ Tracks all your specified metrics
- ✅ Supports A/B testing out of the box
- ✅ Has training pipelines ready for when you have data
- ✅ Follows SOLID principles for easy extension
- ✅ Is fully dockerized
- ✅ Has comprehensive documentation

You can now:
1. Start collecting interaction data
2. Train models as data accumulates
3. Run A/B tests to improve recommendations
4. Scale horizontally as needed

Good luck with your project! 🚀

