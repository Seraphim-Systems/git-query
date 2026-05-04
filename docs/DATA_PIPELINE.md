# Data Pipeline: Ingest -> Normalize -> Vectorize -> Train

This document explains the data-processing pipeline used by Git-Query.

The goal is to make the flow understandable for anyone who needs to extend,
debug, or evaluate the recommender without having to read the full codebase first.

## High-level flow

```text
Raw repository data
        |
        v
Ingestion from API / storage
        |
        v
Normalization and validation
        |
        v
Text construction + deduplication
        |
        v
Embedding generation
        |
        v
Upload to Qdrant
        |
        v
LightGBM reranker training
        |
        v
Model artifacts + checkpoints
```

## 1. Inputs

The pipeline starts from repository records fetched from the backend API.

Typical raw fields include repository metadata such as:

* name
* description
* topics
* language
* stars / forks / update timestamps

## 2. Ingestion

The training pipeline:

* fetches repository data from the API
* processes data in batches
* supports chunked execution for scalability

## 3. Normalization

This stage:

* cleans missing values
* standardizes text fields
* ensures consistent schema

## 4. Text construction

Each repository is transformed into a unified text string combining:

* name
* description
* topics
* language

This improves embedding quality.

## 5. Vectorization

* Text is embedded using a sentence transformer
* Processing is batched for efficiency
* Outputs fixed-size vectors

## 6. Vector upload

Vectors are uploaded to Qdrant:

* batched uploads
* metadata stored alongside embeddings
* avoids duplication using checkpoints

## 7. Checkpointing

Supports resumable runs by storing:

* processed repository IDs
* last completed chunk

## 8. Model training

A LightGBM model:

* learns ranking of repositories
* uses features derived from metadata

## 9. Outputs

* embeddings in Qdrant
* trained model
* logs and metadata

## 10. Extending the pipeline

When adding steps:

1. keep them modular
2. log clearly
3. ensure reproducibility
4. document changes

## 11. Why this matters

This pipeline connects raw data with recommendation quality.

It enables:

* debugging
* reproducibility
* team collaboration

## 12. Related files

* `README.md`
* `training.md`
* `src/recommender/training/unified_pipeline.py`