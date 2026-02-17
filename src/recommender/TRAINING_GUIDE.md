# ML Pipeline Training Guide

Complete guide to fetch data from your server, train models locally, and deploy them.

## Overview

This workflow allows you to:
1. **Fetch data** from your remote server via API
2. **Train ML models** locally on your machine
3. **Store models** locally for testing
4. **Upload embeddings** back to the server (optional)

---

## Prerequisites

- Python 3.11+ with virtual environment activated
- Your server API keys (MongoDB, Qdrant)
- Server URL (e.g., `http://your-server.com` or `http://localhost:8000`)
- Sufficient disk space for data and models (~1-5GB depending on dataset size)

---

## Step 1: Fetch Data from Server

Run the data fetching script:

```bash
python -m src.recommender.scripts.fetch_data_from_server
```

**What it does:**
- Connects to your server API
- Fetches repository data in batches
- Saves data locally to `./data/training/`

**Interactive prompts:**
```
Enter server URL: http://your-server.com
Enter MongoDB API key: your-mongodb-key
Also fetch Qdrant vectors? (y/n): n
Max repositories to fetch (leave empty for all): 1000
```

**Output:**
- `./data/training/repositories_latest.json` - Latest repository data
- `./data/training/repositories_YYYYMMDD_HHMMSS.json` - Timestamped backup
- `./data/training/interactions_latest.json` - User interactions (if available)

**Example:**
```bash
# Fetch all repositories
python -m src.recommender.scripts.fetch_data_from_server

# This will:
# ✓ Check server health
# ✓ List available collections
# ✓ Fetch repositories in batches of 100
# ✓ Save to local JSON files
```

---

## Step 2: Train Models Locally

Once data is fetched, train the embeddings:

```bash
python -m src.recommender.scripts.train_local
```

**What it does:**
- Loads repository data from `./data/training/`
- Generates embeddings using sentence-transformers
- Saves models locally to `./models/`

**Interactive prompts:**
```
Embedding model (default: sentence-transformers/all-MiniLM-L6-v2): 
Batch size (default: 32): 
Proceed with training? (y/n): y
```

**Output:**
- `./models/vectors/repo_embeddings_latest.npy` - Embedding vectors
- `./models/metadata/repo_mapping_latest.json` - Repo ID to index mapping
- `./models/metadata/training_metadata_latest.json` - Training metadata

**Training time:**
- 1,000 repos: ~2-5 minutes
- 10,000 repos: ~15-30 minutes
- Uses GPU if available (CUDA), otherwise CPU

**Example output:**
```
Generating embeddings using sentence-transformers/all-MiniLM-L6-v2
Using device: cuda

Preparing repository texts...
✓ Prepared 1000 texts

Generating embeddings (batch_size=32)...
100%|████████████████████| 32/32 [00:45<00:00]
✓ Generated 1000 embeddings
  Embedding dimension: 384

✓ Saved embeddings to ./models/vectors/repo_embeddings_20260216_143022.npy
✓ Saved mapping to ./models/metadata/repo_mapping_20260216_143022.json
```

---

## Step 3: Test Locally (Optional)

Test the trained models locally without uploading:

```bash
python -m src.recommender.scripts.test_ml_components
```

This verifies all components load correctly.

---

## Step 4: Upload Embeddings to Server (Optional)

If you want to deploy the embeddings to your server's Qdrant instance:

```bash
python -m src.recommender.scripts.upload_embeddings
```

**What it does:**
- Loads locally trained embeddings
- Uploads them to Qdrant via server API
- Batches uploads to respect rate limits

**Interactive prompts:**
```
Enter server URL: http://your-server.com
Enter Qdrant API key: your-qdrant-key
Collection name (default: repositories_embeddings): 
Batch size (default: 100): 
Proceed with upload? (y/n): y
```

**Upload time:**
- 1,000 vectors: ~1-2 minutes
- 10,000 vectors: ~10-15 minutes
- Respects rate limits (100 req/min)

---

## Directory Structure

After running all scripts:

```
git-query/
├── data/
│   └── training/
│       ├── repositories_latest.json       # Latest fetched data
│       ├── repositories_20260216.json     # Timestamped backup
│       └── interactions_latest.json       # User interactions
│
├── models/
│   ├── vectors/
│   │   ├── repo_embeddings_latest.npy    # Embedding vectors
│   │   └── repo_embeddings_20260216.npy  # Timestamped backup
│   │
│   └── metadata/
│       ├── repo_mapping_latest.json      # ID mappings
│       ├── training_metadata_latest.json # Training info
│       └── *.json                        # Timestamped backups
│
└── src/recommender/scripts/
    ├── fetch_data_from_server.py         # Step 1
    ├── train_local.py                    # Step 2
    └── upload_embeddings.py              # Step 4
```

---

## Complete Workflow Example

```bash
# 1. Fetch data from server
python -m src.recommender.scripts.fetch_data_from_server
# Enter your server URL and API keys
# This saves data to ./data/training/

# 2. Train models locally
python -m src.recommender.scripts.train_local
# This generates embeddings and saves to ./models/

# 3. (Optional) Upload to server
python -m src.recommender.scripts.upload_embeddings
# This uploads embeddings back to Qdrant
```

---

## Configuration Files

### `.env` (already created)
Location: `src/recommender/.env`

Contains local configuration for the recommender service.

### API Keys (you need these)
- **MongoDB API Key**: From your server's `APIKEY_MONGODB` environment variable
- **Qdrant API Key**: From your server's `APIKEY_QDRANT` environment variable

---

## Troubleshooting

### "No repository data found"
- Make sure you ran Step 1 (fetch_data_from_server)
- Check that `./data/training/repositories_latest.json` exists

### "Server health check failed"
- Verify server URL is correct
- Check server is running and accessible
- Test with: `curl http://your-server.com/api/health`

### "Out of memory during training"
- Reduce batch_size (try 16 or 8)
- Fetch fewer repositories
- Use a smaller embedding model

### "Rate limit exceeded"
- Wait 1 minute and retry
- Reduce batch size in upload script
- Server allows 100 requests/minute

---

## Next Steps After Training

1. **Test recommendations locally:**
   - Start local recommender: `python -m src.recommender`
   - Access at: `http://localhost:8095`
   - Test endpoint: `http://localhost:8095/health`

2. **Integrate with your app:**
   - Use the trained embeddings for semantic search
   - Connect to local Qdrant or uploaded embeddings on server

3. **Retrain periodically:**
   - Re-run fetch + train as your data grows
   - Models are versioned by timestamp

---

## API Routes Used

From your server's `API_ROUTES.md`:

- `GET /api/health` - Check server health
- `GET /api/mongodb/collections` - List collections
- `POST /api/mongodb/collections/repositories/query` - Fetch repositories
- `POST /api/qdrant/collections/{collection}/points` - Upload embeddings

---

## Questions?

- Check `src/recommender/README.md` for ML pipeline details
- See `docs/API_ROUTES.md` for server API documentation
- Run scripts with `-h` for help (where applicable)

---

**Happy Training! 🚀**
"""Fetch repository data from the server API in batches and store locally."""

import requests
import json
import os
from pathlib import Path
from typing import List, Dict, Any
import time
from datetime import datetime


class DataFetcher:
    """Fetch data from git-query server API."""
    
    def __init__(
        self,
        base_url: str,
        mongodb_api_key: str,
        qdrant_api_key: str = None,
        output_dir: str = "./data"
    ):
        self.base_url = base_url.rstrip('/')
        self.mongodb_api_key = mongodb_api_key
        self.qdrant_api_key = qdrant_api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.headers_mongodb = {
            "Authorization": f"Bearer {mongodb_api_key}",
            "Content-Type": "application/json"
        }
        
        if qdrant_api_key:
            self.headers_qdrant = {
                "Authorization": f"Bearer {qdrant_api_key}",
                "Content-Type": "application/json"
            }
    
    def check_health(self) -> Dict[str, Any]:
        """Check server health status."""
        try:
            response = requests.get(f"{self.base_url}/api/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Health check failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def list_collections(self) -> List[str]:
        """List all MongoDB collections."""
        try:
            response = requests.get(
                f"{self.base_url}/api/mongodb/collections",
                headers=self.headers_mongodb
            )
            response.raise_for_status()
            data = response.json()
            return data.get("collections", [])
        except Exception as e:
            print(f"Failed to list collections: {e}")
            return []
    
    def fetch_repositories_batch(
        self,
        batch_size: int = 100,
        skip: int = 0,
        filter_query: Dict = None,
        sort_by: Dict = None
    ) -> Dict[str, Any]:
        """Fetch a batch of repositories from MongoDB."""
        query = {
            "filter": filter_query or {},
            "limit": batch_size,
            "skip": skip,
        }
        
        if sort_by:
            query["sort"] = sort_by
        
        try:
            response = requests.post(
                f"{self.base_url}/api/mongodb/collections/repositories/query",
                headers=self.headers_mongodb,
                json=query
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Failed to fetch batch (skip={skip}): {e}")
            return {"count": 0, "documents": []}
    
    def fetch_all_repositories(
        self,
        batch_size: int = 100,
        max_repos: int = None,
        filter_query: Dict = None,
        sort_by: Dict = None
    ) -> List[Dict]:
        """Fetch all repositories in batches."""
        all_repos = []
        skip = 0
        total_fetched = 0
        
        print(f"\n{'='*60}")
        print(f"Fetching repositories from server...")
        print(f"Batch size: {batch_size}")
        if max_repos:
            print(f"Max repositories: {max_repos}")
        print(f"{'='*60}\n")
        
        while True:
            # Check if we've reached max
            if max_repos and total_fetched >= max_repos:
                break
            
            # Adjust batch size for last batch
            current_batch_size = batch_size
            if max_repos:
                remaining = max_repos - total_fetched
                current_batch_size = min(batch_size, remaining)
            
            print(f"Fetching batch: skip={skip}, limit={current_batch_size}")
            
            result = self.fetch_repositories_batch(
                batch_size=current_batch_size,
                skip=skip,
                filter_query=filter_query,
                sort_by=sort_by
            )
            
            count = result.get("count", 0)
            documents = result.get("documents", [])
            
            if not documents:
                print("No more documents to fetch.")
                break
            
            all_repos.extend(documents)
            total_fetched += len(documents)
            print(f"✓ Fetched {len(documents)} repositories (total: {total_fetched})")
            
            # If we got fewer documents than requested, we're done
            if count < current_batch_size:
                print("Reached end of collection.")
                break
            
            skip += current_batch_size
            
            # Rate limiting - be nice to the server
            time.sleep(0.5)
        
        print(f"\n{'='*60}")
        print(f"✓ Total repositories fetched: {total_fetched}")
        print(f"{'='*60}\n")
        
        return all_repos
    
    def fetch_user_interactions(
        self,
        batch_size: int = 100,
        max_interactions: int = None
    ) -> List[Dict]:
        """Fetch user interactions for training."""
        return self._fetch_collection(
            "user_interactions",
            batch_size=batch_size,
            max_items=max_interactions
        )
    
    def _fetch_collection(
        self,
        collection_name: str,
        batch_size: int = 100,
        max_items: int = None,
        filter_query: Dict = None
    ) -> List[Dict]:
        """Generic method to fetch any collection."""
        all_items = []
        skip = 0
        total_fetched = 0
        
        print(f"\nFetching {collection_name}...")
        
        while True:
            if max_items and total_fetched >= max_items:
                break
            
            current_batch_size = batch_size
            if max_items:
                remaining = max_items - total_fetched
                current_batch_size = min(batch_size, remaining)
            
            query = {
                "filter": filter_query or {},
                "limit": current_batch_size,
                "skip": skip,
            }
            
            try:
                response = requests.post(
                    f"{self.base_url}/api/mongodb/collections/{collection_name}/query",
                    headers=self.headers_mongodb,
                    json=query
                )
                response.raise_for_status()
                result = response.json()
                
                documents = result.get("documents", [])
                if not documents:
                    break
                
                all_items.extend(documents)
                total_fetched += len(documents)
                print(f"✓ Fetched {len(documents)} items from {collection_name} (total: {total_fetched})")
                
                if len(documents) < current_batch_size:
                    break
                
                skip += current_batch_size
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error fetching {collection_name}: {e}")
                break
        
        print(f"✓ Total {collection_name} fetched: {total_fetched}\n")
        return all_items
    
    def save_to_json(self, data: List[Dict], filename: str):
        """Save data to JSON file."""
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved {len(data)} items to {filepath}")
    
    def load_from_json(self, filename: str) -> List[Dict]:
        """Load data from JSON file."""
        filepath = self.output_dir / filename
        if not filepath.exists():
            print(f"File not found: {filepath}")
            return []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ Loaded {len(data)} items from {filepath}")
        return data


def main():
    """Main function to fetch data from server."""
    
    # Configuration
    BASE_URL = input("Enter server URL (e.g., http://your-server.com or http://localhost:8000): ").strip()
    MONGODB_API_KEY = input("Enter MongoDB API key: ").strip()
    
    # Optional: fetch vectors too
    fetch_vectors = input("Also fetch Qdrant vectors? (y/n): ").strip().lower() == 'y'
    QDRANT_API_KEY = None
    if fetch_vectors:
        QDRANT_API_KEY = input("Enter Qdrant API key: ").strip()
    
    # Batch settings
    BATCH_SIZE = 100
    MAX_REPOS = int(input("Max repositories to fetch (leave empty for all): ").strip() or "0") or None
    
    # Initialize fetcher
    fetcher = DataFetcher(
        base_url=BASE_URL,
        mongodb_api_key=MONGODB_API_KEY,
        qdrant_api_key=QDRANT_API_KEY,
        output_dir="./data/training"
    )
    
    # Check health
    print("\n" + "="*60)
    print("Checking server health...")
    print("="*60)
    health = fetcher.check_health()
    print(json.dumps(health, indent=2))
    
    if health.get("status") != "healthy":
        print("\n⚠ Warning: Server may not be fully healthy")
        proceed = input("Continue anyway? (y/n): ").strip().lower()
        if proceed != 'y':
            return
    
    # List collections
    print("\n" + "="*60)
    print("Available collections:")
    print("="*60)
    collections = fetcher.list_collections()
    for col in collections:
        print(f"  - {col}")
    
    # Fetch repositories
    print("\n" + "="*60)
    print("Fetching repository data...")
    print("="*60)
    
    repositories = fetcher.fetch_all_repositories(
        batch_size=BATCH_SIZE,
        max_repos=MAX_REPOS,
        sort_by={"stars": -1}  # Get most popular first
    )
    
    # Save repositories
    if repositories:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fetcher.save_to_json(repositories, f"repositories_{timestamp}.json")
        fetcher.save_to_json(repositories, "repositories_latest.json")
    
    # Fetch user interactions (if available)
    if "user_interactions" in collections:
        print("\n" + "="*60)
        print("Fetching user interactions...")
        print("="*60)
        interactions = fetcher.fetch_user_interactions(batch_size=BATCH_SIZE)
        if interactions:
            fetcher.save_to_json(interactions, f"interactions_{timestamp}.json")
            fetcher.save_to_json(interactions, "interactions_latest.json")
    
    print("\n" + "="*60)
    print("✓ Data fetch complete!")
    print("="*60)
    print(f"\nData saved to: {fetcher.output_dir}")
    print(f"  - Repositories: {len(repositories)}")
    print("\nNext steps:")
    print("  1. Run: python -m src.recommender.scripts.train_local")
    print("  2. Models will be saved locally")
    print("="*60)


if __name__ == "__main__":
    main()

