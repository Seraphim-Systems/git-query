# Training Pipeline - Docker Setup

This directory contains a fully containerized training pipeline that fetches repository data from your API and trains embedding models.

## Overview

**What it does:**
1. ✅ Fetches repository data from MongoDB API (remote server)
2. ✅ Trains embedding model using sentence-transformers
3. ✅ Saves trained model to persistent volume
4. ✅ Caches data locally for future runs
5. ✅ Only retrains when new data is available

**This is NOT the reranker** - this trains the initial embedding model that comes before the reranker in your pipeline.

## Quick Start

### 1. Create Environment File

**IMPORTANT**: Copy the example and fill in your actual API credentials:

```bash
# From the infrastructure/docker directory
cp .env.training.example .env

# Edit .env and set your actual values:
# - API_BASE_URL (your API server URL)
# - APIKEY_MONGODB (your API key)
```

**Example .env file structure:**

```bash
# API Configuration (REQUIRED - Replace with your actual values)
API_BASE_URL=https://your-actual-api-server.com
APIKEY_MONGODB=your_actual_api_key_here

# Training Configuration
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
BATCH_SIZE=32
FETCH_BATCH_SIZE=100
SKIP_IF_NO_NEW_DATA=true

# Optional: Limit number of repos for testing
# MAX_REPOS=1000
```

⚠️ **Security Note**: Never commit your `.env` file with actual credentials to version control!

### 2. Run Training

```bash
# Make sure you're in the infrastructure/docker directory
cd infrastructure/docker

# Run the training pipeline
docker-compose -f docker-compose.training.yml up --build
```

The container will:
- Read credentials from your `.env` file
- Fetch data from your API
- Train the embedding model
- Save results to a Docker volume

### 3. Verify Results

```bash
# Check what was created
docker run --rm -v git-query-training-models:/models alpine ls -lR /models
```

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_BASE_URL` | **YES** | - | Your API server URL (no default for security) |
| `APIKEY_MONGODB` | **YES** | - | Your API key (no default for security) |
| `EMBEDDING_MODEL` | No | `sentence-transformers/all-MiniLM-L6-v2` | Model to use |
| `BATCH_SIZE` | No | `32` | Embedding batch size |
| `FETCH_BATCH_SIZE` | No | `100` | API fetch batch size |
| `SKIP_IF_NO_NEW_DATA` | No | `true` | Skip training if no new data |
| `MAX_REPOS` | No | - | Limit repos (for testing) |
| `LOG_LEVEL` | No | `INFO` | Logging level |

## Security Best Practices

1. **Never hardcode credentials** in code or Docker files
2. **Use `.env` files** for local development
3. **Add `.env` to `.gitignore`** to prevent accidental commits
4. **Use secrets management** for production (Docker secrets, Kubernetes secrets, etc.)
5. **Rotate API keys** regularly

## Troubleshooting

### "Missing required environment variables" error

Make sure you've created the `.env` file with actual values:

```bash
# Check if .env exists
ls -la .env

# Verify it has the required variables
cat .env | grep API_BASE_URL
cat .env | grep APIKEY_MONGODB
```

### Container fails to start

Check the logs:
```bash
docker-compose -f docker-compose.training.yml logs
```

Common issues:
- Missing or empty `.env` file
- Wrong API credentials
- Network connectivity issues

## Running on Other Machines

To run this on different hardware:

1. **Copy these files:**
   ```
   infrastructure/docker/
   src/recommender/
   ```

2. **Create `.env` with your credentials**

3. **Run:**
   ```bash
   docker-compose -f docker-compose.training.yml up --build
   ```

The `.env` file stays on each machine - never transfer it over insecure channels!

## Scheduled Training

### Daily Cron Job

```bash
# Add to crontab (update paths and credentials)
0 2 * * * cd /path/to/git-query/infrastructure/docker && \
  docker-compose -f docker-compose.training.yml up --build
```

Make sure the `.env` file exists in the docker directory!

---

For more details, see `src/recommender/TRAINING_SETUP_SUMMARY.md`

