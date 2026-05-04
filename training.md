# Combined Training Pipeline (Embeddings + Model)

This guide explains how to run the full retraining pipeline using Docker.

For a detailed explanation of how data flows through the system (ingestion, normalization, vectorization, and training), see:

 `docs/DATA_PIPELINE.md`

---

## Overview

The training pipeline has two main stages:

1. **Embedding generation and indexing**
2. **LightGBM reranker training**

The goal is to transform raw repository data into:

* semantic embeddings stored in Qdrant
* a trained ranking model for improved recommendations

---

## What this pipeline does

The training container runs:

```bash
python -m training.retrain_pipeline
```

This executes:

* Step 1: Fetch repository data from API
* Step 2: Normalize and construct text representations
* Step 3: Generate embeddings
* Step 4: Upload embeddings to Qdrant
* Step 5: Train LightGBM reranker

---

## Pipeline modes

The system supports two execution modes:

### 1. One-shot mode (`run()`)

* Processes all repositories in one pass
* Generates a single embeddings file
* Simpler but less scalable

### 2. Chunked mode (`run_chunked()`)

* Processes repositories in chunks/windows
* Deduplicates across batches
* Uploads embeddings incrementally
* Saves checkpoints after each chunk

Recommended for large datasets

---

## Prerequisites

* Docker installed and running
* Repository cloned locally
* Environment variables configured

---

## Environment configuration

Use:

```bash
infrastructure/docker/.env
```

### Required variables

* `API_BASE_URL`
* `APIKEY_MONGODB`
* `APIKEY_QDRANT`

### Recommended settings (for stability and visibility)

```bash
CHUNK_SIZE=5000
FETCH_BATCH_SIZE=200
N_WORKERS=1
```

### Optional flags

* Disable upload to Qdrant (for testing):

```bash
SKIP_QDRANT_UPLOAD=true
```

---

## Running the pipeline

Run the training container:

```bash
docker compose \
  -f infrastructure/docker/docker-compose.base.yml \
  -f infrastructure/docker/docker-compose.training.yml \
  --env-file infrastructure/docker/.env \
  run -d --name training_run training
```

---

## Checkpointing and resumability

The pipeline supports resumable execution.

Key features:

* `models/metadata/repo_mapping_latest.json` stores processed repositories
* Each chunk saves progress after completion
* Prevents reprocessing already indexed data
* Enables recovery after crashes

---

## Outputs

After a successful run, the pipeline produces:

* Embeddings stored in Qdrant
* Trained LightGBM model
* Metadata and mapping files
* Logs for reproducibility

---

## Tips for development

* Use smaller `CHUNK_SIZE` and `FETCH_BATCH_SIZE` for faster iteration
* Disable Qdrant upload when testing locally
* Run chunked mode for better visibility of progress
* Monitor logs to debug failures early

---

## Extending the pipeline

When modifying or adding new steps:

1. Keep steps modular
2. Add logging before and after execution
3. Ensure reproducibility
4. Avoid breaking checkpointing
5. Document changes in `docs/DATA_PIPELINE.md`

---

## Related documentation

* `docs/DATA_PIPELINE.md`
* `docs/MODEL_EXPLANATION.md`
* `README.md`
* `src/recommender/training/unified_pipeline.py`
