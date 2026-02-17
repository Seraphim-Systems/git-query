# Training Quickstart Guide

## Overview
This guide walks you through training the recommendation model locally using data from your server.

## Prerequisites
- Python environment with dependencies installed
- Access to MongoDB API (set API_BASE_URL and APIKEY_MONGODB in your .env file)

## Training Pipeline

### Step 1: Fetch Data from Server
```bash
python -m src.recommender.scripts.fetch_data_from_server
```

This will:
- Connect to your MongoDB API
- Fetch repositories in batches
- Save data to `./data/training/repositories_latest.json`
- Show data summary (languages, stars, etc.)

**Note:** Current server has limited data (~5 repos showing). You may want to scrape more repositories before training.

### Step 2: Train the Model
```bash
python -m src.recommender.scripts.train_local
```

This will:
- Load repository data
- Generate embeddings using Sentence Transformers
- Save checkpoints every 100 repositories
- Store trained models in `src/recommender/models/`

**Features:**
- ✅ **Checkpointing**: Training saves progress every 100 repos
- ✅ **Resume capability**: If interrupted, just run again and it will resume
- ✅ **GPU support**: Automatically uses CUDA if available
- ✅ **Progress tracking**: Shows real-time progress

### Model Output Structure
```
src/recommender/models/
├── checkpoints/          # Training checkpoints (auto-cleanup after completion)
├── vectors/              # Embedding vectors (.npy files)
│   ├── repo_embeddings_latest.npy
│   └── repo_embeddings_20260216_143000.npy
├── metadata/             # Training metadata & mappings
│   ├── repo_mapping_latest.json
│   ├── training_metadata_latest.json
│   └── ...
└── embeddings/           # Future: Fine-tuned models
```

## Quick Training Run

For a quick test with current data:

```bash
# 1. Fetch all available data
python -m src.recommender.scripts.fetch_data_from_server

# When prompted:
# - Server URL: https://gitquery.davidhoerz.com (or press Enter)
# - API key: apikey (or your key)
# - Batch size: 100 (default)
# - Max repos: leave empty for all

# 2. Train the model
python -m src.recommender.scripts.train_local

# When prompted:
# - Embedding model: press Enter for default
# - Batch size: 32 (default)
# - Resume from checkpoint: y (default)
# - Proceed: y
```

## Training Interruption

If training is interrupted (Ctrl+C or crash):
1. Don't worry! Progress is saved in checkpoints
2. Simply run the training command again
3. Answer 'y' to "Resume from checkpoint?"
4. Training continues from where it stopped

## Recommendations for Better Training

### Current Limitation
The server currently has very few repositories (~5 visible). For effective training, you need:
- **Minimum**: 1,000+ repositories
- **Recommended**: 10,000+ repositories
- **Ideal**: 100,000+ repositories

### Before Training
1. Run the scraper to collect more repositories
2. Or wait until more data is available on the server
3. Check data quality with the fetch script first

### Training Configuration

**For small datasets (<1,000 repos):**
```
Batch size: 16
Model: all-MiniLM-L6-v2 (default, fast)
```

**For medium datasets (1,000-10,000 repos):**
```
Batch size: 32
Model: all-MiniLM-L6-v2
```

**For large datasets (10,000+ repos):**
```
Batch size: 64
Model: all-mpnet-base-v2 (better quality, slower)
```

## Next Steps After Training

1. **Test locally**: Use the trained embeddings for recommendations
2. **Upload to Qdrant**: `python -m src.recommender.scripts.upload_embeddings`
3. **Start API**: `python -m src.recommender`
4. **Evaluate**: `python -m src.recommender.scripts.evaluate`

## Troubleshooting

### No data fetched
- Check API key is correct
- Verify server URL is accessible
- Check if repositories collection has data

### Training fails
- Check if `./data/training/repositories_latest.json` exists
- Verify enough disk space for embeddings
- Check CUDA/GPU drivers if using GPU

### Out of memory
- Reduce batch size (try 16 or 8)
- Use CPU instead of GPU
- Process fewer repositories at a time

## Files Created

After successful training:
- `src/recommender/models/vectors/repo_embeddings_latest.npy` - Embedding vectors
- `src/recommender/models/metadata/repo_mapping_latest.json` - Repo ID to index mapping
- `src/recommender/models/metadata/training_metadata_latest.json` - Training config

# Set up your API credentials in .env first:
# - API_BASE_URL=https://your-server.com
# - APIKEY_MONGODB=your_api_key
