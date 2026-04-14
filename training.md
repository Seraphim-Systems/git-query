# Combined Training Pipeline (Embeddings + Model)

This guide explains how to run the full retraining pipeline in Docker:

1. Embedding indexing + upload to Qdrant
2. Model training (LightGBM reranker)

## What this pipeline does

The training container runs `python -m training.retrain_pipeline`, which executes:

- **Step 1:** Embedding indexing pipeline
- **Step 2:** LightGBM reranker training

Embeddings are uploaded to Qdrant automatically unless explicitly disabled.

## Prerequisites

- Docker is installed and running.
- You are in the repo root.
- `infrastructure/docker/.env` is configured.

## Recommended environment file

Use `infrastructure/docker/.env` for this flow.

Minimum important variables:

- `API_BASE_URL` (gateway URL used by training)
- `APIKEY_MONGODB`
- `APIKEY_QDRANT`
- `UPLOAD_BATCH_SIZE`
- `SKIP_QDRANT_UPLOAD` (optional; default behavior is upload enabled)

Stability/progress settings (recommended for visible progress):

- `CHUNK_SIZE=5000`
- `FETCH_BATCH_SIZE=200`
- `N_WORKERS=1`

## Run the combined pipeline

```powershell
docker compose -f infrastructure/docker/docker-compose.base.yml -f infrastructure/docker/docker-compose.training.yml --env-file infrastructure/docker/.env run -d --name training_run training
```

## Monitor progress

```powershell
docker logs -f training_run
```

You should see:

- `=== Step 1: Embedding indexing ===`
- chunk progress (`CHUNK x/y`)
- embedding lines (`Embedding ... texts`)
- upload lines (`Uploading ... points to Qdrant` and `Uploaded ...`)
- later: `=== Step 2: LightGBM reranker ===`

## Quick health checks

Container state:

```powershell
docker ps -a --filter "name=training_run" --format "table {{.Names}}\t{{.Status}}"
```

Recent Qdrant upload evidence:

```powershell
docker logs --tail 200 training_run | Select-String -Pattern "Qdrant|Upload|Uploaded|upload"
```

Runtime usage:

```powershell
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" training_run
```

## Notes on `unhealthy` status

For this training container, `unhealthy` can appear during long runs due to healthcheck behavior. If logs show active chunking/embedding/uploading, the pipeline is still progressing.

## Stop / cleanup

Stop and remove only this training run:

```powershell
docker rm -f training_run
```

Remove all containers (dangerous, global cleanup):

```powershell
$all = docker ps -aq; if ($all) { docker rm -f $all }
```

## Common pitfalls

- Using the wrong env file (`.env` at repo root instead of `infrastructure/docker/.env`).
- Placeholder `API_BASE_URL` (for example `your-gateway-url.com`) causing fetch failures.
- Setting `SKIP_QDRANT_UPLOAD=true` and expecting vectors in Qdrant.

## Expected success pattern in logs

A healthy run usually repeats this cycle per chunk:

- `Fetched ... repos`
- `Embedding ... texts`
- `Uploaded .../...`
- mapping save/update

After embedding phase finishes, it proceeds to LightGBM training (Step 2).
