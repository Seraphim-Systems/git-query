# ML Ranker Model & Recommendation Architecture

<!-- Source: src/recommender/engines/hybrid.py, src/recommender/services/reranker_service.py, src/recommender/training/evaluator.py, src/recommender/services/registry_service.py -->

This document explains the architecture, engineering infrastructure, and operational workflows for the GitHub Query recommendation engine. It is intended for ML Engineers and Data Scientists working on improving recommendation quality.

## Two-Stage Retrieval Architecture

The system employs a classic two-stage retrieval pattern to balance low latency with high precision.

### 1. Stage 1: Retrieval (Bi-Encoder + Keyword)
The goal of the first stage is to quickly narrow down millions of repositories to a candidate set of ~100-200 relevant items.

*   **Semantic Search (Bi-Encoder)**: We use `all-MiniLM-L6-v2` (default) to generate vector embeddings for both queries and repository metadata (name, description, topics). These are stored in **Qdrant**. At query time, we perform a cosine similarity search.
*   **Keyword Search**: Parallel to semantic search, we use MongoDB's text search to find exact matches for specific technologies or library names that might be missed by embeddings.
*   **Score Fusion**: Results from both sources are combined using **Reciprocal Rank Fusion (RRF)** to produce a unified candidate list.

### 2. Stage 2: Ranking (Cross-Encoder)
The second stage uses a more computationally expensive but significantly more accurate model to score the top candidates.

*   **Precise Scoring**: A **Cross-Encoder** (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`) processes `(query, repository_text)` pairs. Unlike Bi-Encoders, Cross-Encoders perform full self-attention over both the query and the document simultaneously, capturing complex interactions.
*   **Final Ordering**: The candidates are re-sorted based on the Cross-Encoder's output score, and the top $K$ results are returned to the user.

---

## ML Engineering Infrastructure

### Checkpointing
During training, the `RerankerTrainer` saves intermediate model weights to ensure progress isn't lost and to allow for early stopping/model selection.

*   **Location**: `models/checkpoints/{variant}/epoch_{n}`
*   **Retention**: By default, the system keeps the last 3 checkpoints.
*   **Structure**: Each checkpoint is a full `sentence-transformers` compatible directory that can be loaded directly for evaluation or inference.

### Evaluation Pipeline
We evaluate model performance using historical user interactions (clicks and saves) as ground truth.

*   **Metrics**:
    *   **NDCG (Normalized Discounted Cumulative Gain)**: Measures the quality of the ranking, rewarding relevant items appearing higher in the list.
    *   **MRR (Mean Reciprocal Rank)**: Measures how deep a user has to look to find the first relevant result.
    *   **Precision@K / Recall@K**: Standard set-based retrieval metrics.
*   **Historical Clicks**: The `RecommenderEvaluator` replays historical queries and compares the model's top $K$ predictions against the repositories the user actually interacted with.

### Model Registry
The Registry manages the lifecycle of models from training to production.

*   **Storage**: Model metadata is stored in the `models` collection in MongoDB.
*   **Promotion**: Models are initially registered with `status: "candidate"`. To promote a model to production, the `ModelRegistryService.promote_model(model_id)` method:
    1.  Sets the new model's `is_active` flag to `True`.
    2.  Deactivates the previous active model of the same type and variant.
    3.  Updates the system state so the Gateway starts routing requests to the new artifact.

---

## Data Scientist's Guide

### Plugging in a New Base Model
To experiment with a new architecture (e.g., DeBERTa for ranking or a larger MPNet for embeddings), follow these steps:

1.  **Update Configuration**: Modify `src/recommender/config.py` or set environment variables:
    ```bash
    # For Retrieval (Bi-Encoder)
    EMBEDDING_MODEL_NAME="sentence-transformers/all-mpnet-base-v2"
    
    # For Ranking (Cross-Encoder)
    CROSS_ENCODER_MODEL_NAME="cross-encoder/ms-marco-electra-base"
    ```
2.  **Retrain**: Run the training script with the new base model. The trainer will automatically pull the weights from HuggingFace and begin fine-tuning on our local interaction data.
3.  **Validate**: Ensure the `ModelMetadata` in MongoDB correctly reflects the new `base_model` in the `hyperparameters` field.

### Interpreting Evaluation Reports
After an evaluation run (via `scripts/evaluate.py`), reports are saved in `models/eval/`.

**Example Report (`models/eval/report_20240216.json`):**
```json
{
  "precision_at_k": { "5": 0.42, "10": 0.35 },
  "ndcg_at_k": { "10": 0.58 },
  "mrr": 0.61,
  "metadata": {
    "model_id": "reranker_v1_1708080000",
    "eval_set_size": 5000
  }
}
```

*   **High MRR but low Recall@K**: Suggests the model is good at finding *one* relevant item quickly, but misses other potentially useful repositories.
*   **Low NDCG**: Indicates that while relevant items are in the top $K$, they are not correctly ordered (e.g., the most relevant item is at rank 10 instead of rank 1).
