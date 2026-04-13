# Git-Query — Recommender + MCP + Training

A compact guide to the Git-Query repository: how the recommender, MCP server, client, and training pipeline fit together, how to run them locally (Docker), where trained models are stored, and common troubleshooting tips.

This README intentionally avoids any secrets — keep API keys and credentials in a local `.env` file.

Table of contents

- Overview
- Architecture
- Key components
- Running locally (Docker / compose)
- Training pipeline (fetch → embed → upload) and checkpointing
- Where trained models live and how to retrieve them
- Important environment variables (.env.example)
- Troubleshooting & common errors
- Next steps / suggestions

## Overview

Git-Query is a modular system to recommend GitHub repositories using a hybrid retrieval + embedding approach. The main pieces in this repository are:

- Recommender service (HTTP API, default port 8095)
- MCP (Model Context Protocol) server — tool router that wraps the recommender and exposes tools to LLM agents (port 8090)
- Client — interactive CLI that runs the LLM agent and calls MCP
- Training pipeline — containerized script that fetches repository data from a remote API, generates embeddings with a Sentence-Transformer, uploads vectors to Qdrant, and saves model artifacts.

## Architecture (dataflow)

- Client (LLM agent) ↔ MCP Server (tools) ↔ Recommender Service ↔ (MongoDB, Redis, Qdrant)
- Training container: calls the remote API (MongoDB HTTP API exposed by your server) → fetches batches → computes embeddings → uploads to Qdrant and writes model artifacts to `models/`.

In Docker compose stacks the internal hostnames are used: `recommender:8095` and `mcp-server:8090`.

## Key components (brief)

- src/recommender: models, embedding service, registry service and training utilities.
- src/mcp: FastAPI server exposing `/tools/list` and `/tools/execute`. Tools in `src/mcp/tools` forward requests to the recommender.
- src/client: interactive chat client using `pydantic_ai` and an LLM provider; agent tools call the MCP client which posts to MCP.
- infrastructure/docker: Dockerfiles and compose fragments for each component.

## Running locally (Docker)

Recommended: use Docker Compose stacks under `infrastructure/docker`. The refactored structure follows SOLID principles:

**Key files:**

- `docker-compose.base.yml` - shared volumes and networks
- `docker-compose.databases.yml` - MongoDB, Redis, Qdrant
- `docker-compose.app.yml` - Gateway, Recommender, MCP Server, Web UI
- `docker-compose.dev.yml` - development overrides (port bindings, debug logging)
- `docker-compose.prod.yml` - production overrides (no ports, persistent volumes)

**Examples:**

Start full stack (dev):

```bash
docker-compose \
  -f infrastructure/docker/docker-compose.base.yml \
  -f infrastructure/docker/docker-compose.databases.yml \
  -f infrastructure/docker/docker-compose.app.yml \
  -f infrastructure/docker/docker-compose.mlflow.yml \
  -f infrastructure/docker/docker-compose.dev.yml \
  up
```

Start MCP Server (with client for interactive testing):

```bash
docker-compose \
  -f infrastructure/docker/docker-compose.base.yml \
  -f infrastructure/docker/docker-compose.mcp.yml \
  up --build
```

Start recommender only:

```bash
docker-compose \
  -f infrastructure/docker/docker-compose.base.yml \
  -f infrastructure/docker/docker-compose.databases.yml \
  -f infrastructure/docker/docker-compose.app.yml \
  up recommender
```

**Notes:**

- Compose files read environment variables from `.env` (or the OS env). Do not commit secrets.
- Services communicate over `git-query-internal-network`
- Development mode exposes ports to host; production mode requires reverse proxy
- Default ports: Gateway (80), Recommender (8095), MCP Server (8090), Web UI (8080)

## Training pipeline (fetch → embed → upload)

A containerized training script exists at `src/recommender/training/unified_pipeline.py`. It supports two modes:

1. One-shot `run()` — fetch all repos (batched), compute embeddings in one pass, save a single embeddings `.npy` and metadata files.
2. Chunked `run_chunked()` — recommended for large datasets: fetch a large dataset in chunks/windows, dedupe, embed chunk, upload chunk to Qdrant, and checkpoint mapping after each chunk.

Checkpointing features already present:

- `models/metadata/repo_mapping_latest.json` — mapping of repo_id → index; used to skip already-processed repos on subsequent runs.
- During chunked run the pipeline saves mapping checkpoints after each chunk and uploads embeddings incrementally to Qdrant. That makes the pipeline resumable across runs and robust to crashes.

To run training container (example):

```cmd
REM Make sure .env has API_BASE_URL and APIKEY_MONGODB set
docker-compose -f infrastructure\docker\docker-compose.base.yml -f infrastructure\docker\docker-compose.databases.yml -f infrastructure\docker\docker-compose.app.yml -f infrastructure\docker\docker-compose.mlflow.yml -f infrastructure\docker\docker-compose.training.yml up --build training
```

Server MLflow deployment details are documented in `docs/MLFLOW_DEPLOYMENT.md`.

For quick local tests where you don't want to upload to Qdrant set:

- SKIP_QDRANT_UPLOAD=true

And to reduce runtime set smaller CHUNK_SIZE and FETCH_BATCH_SIZE while testing.

## Where trained models live & how to access them

Inside the container the training script writes artifacts to `/app/models` (vectors, metadata, checkpoints).

Compose maps `/app/models` to the named Docker volume `model-artifacts` by default. To access artifacts from host:

- Option A (recommended for development): change the compose volume mapping to a host bind-mount (example in `docker-compose.reco.yml` replace `model-artifacts:/app/models` with `./models:/app/models`) so trained models appear under `./models` on the host.

- Option B: copy out files after a run with `docker cp`:

```cmd
REM copy from a finished training container named git-query-training
docker cp git-query-training:/app/models ./models_from_container
```

- Option C: run a temporary container that mounts the named volume and inspect files.

## Important environment variables (.env.example)

Create a `.env` file at repository root (do NOT commit it). Minimal recommended entries:

```text
# API endpoints
API_BASE_URL=https://your-server.example.com

# Keys (keep secret)
APIKEY_MONGODB=replace_with_mongodb_api_key
APIKEY_QDRANT=replace_with_qdrant_key
APP_OPENAI_API_KEY=replace_with_openai_key

# Training tuning
USE_CHUNKED_PIPELINE=true
CHUNK_SIZE=100000
FETCH_BATCH_SIZE=500
BATCH_SIZE=32
SKIP_QDRANT_UPLOAD=false

# Service URLs (override for local docker)
RECOMMENDER_URL=http://recommender:8095
MCP_SERVER_URL=http://mcp-server:8090

# Optional: if you want the client to use a specific model
MODEL_NAME=gpt-4o
```

## Troubleshooting & common issues

- "MCP / Recommender unreachable": When running containers manually with `docker run`, internal DNS names like `recommender` won't resolve. Prefer running with compose so internal hostnames work, or use `host.docker.internal` to reach host services from within container.

- "OpenAI 401 / missing API key": Ensure `APP_OPENAI_API_KEY` or `OPENAI_API_KEY` is present in the environment passed to the client or MCP containers.

- "Agent / pydantic_ai errors (e.g. AgentRunResult has no attribute 'data' or unexpected api_key arg)": This usually indicates a version mismatch between installed `pydantic_ai` (or its OpenAI provider) and the code. Pin expected package versions in `requirements.txt` or update the code to match the installed library's API.

- "docker-compose: network declared as external not found": create the Docker network manually or remove `external: true` from compose fragments so Compose creates networks automatically.

- Model artifacts not visible on host: the compose stack uses named volumes; switch to a host bind-mount during development to make artifacts easily accessible.

## Next steps & suggestions

- For development, mount `./models:/app/models` in the training/recommender compose setup so you can inspect outputs easily.
- Pin Python package versions (especially `pydantic_ai`, `sentence-transformers`, `httpx`) in `requirements.txt` to avoid runtime API mismatches across machines.
- Run a quick small-scale training run with `SKIP_QDRANT_UPLOAD=true` and small `CHUNK_SIZE` to validate the pipeline and artifact writes.
- If you need precise resumption during embedding of a single large chunk, consider adding partial-embedding saves and a merge step; the current chunked pipeline already provides robust per-chunk checkpointing.

If you'd like, I can now:

- Produce a `.env.example` file with the fields above (no secrets),
- Modify `docker-compose` to bind-mount `./models` for development, or
- Pin the `pydantic_ai` version and update `requirements.txt` to a matching set.

Tell me which of those you'd like me to do next.
