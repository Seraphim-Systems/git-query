"""Unified training pipeline: Fetch data from server and train embeddings.

This script:
1. Fetches repository data from MongoDB API (only new/updated repos)
2. Trains embedding model using sentence-transformers
3. Saves trained embeddings and metadata to models directory
4. Designed to run in Docker container
"""

import json
import os
import time
import requests
import torch
import numpy as np
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from sentence_transformers import SentenceTransformer
import logging

try:
    from ..scripts.upload_embeddings import EmbeddingUploader
except ImportError:
    # When run as `python -m training.unified_pipeline` inside Docker,
    # the scripts package lives at /app/scripts (sibling directory).
    from scripts.upload_embeddings import EmbeddingUploader

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnifiedTrainingPipeline:
    """Fetch data from API and train embeddings in one pipeline."""

    def __init__(
        self,
        api_base_url: str,
        api_key: str,
        models_dir: str = "/app/models",
        data_cache_dir: str = "/app/training_data"
    ):
        self.api_base_url = api_base_url.rstrip('/')
        self.api_key = api_key
        self.models_dir = Path(models_dir)
        self.data_cache_dir = Path(data_cache_dir)

        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.data_cache_dir.mkdir(parents=True, exist_ok=True)
        (self.models_dir / "vectors").mkdir(exist_ok=True)
        (self.models_dir / "metadata").mkdir(exist_ok=True)
        (self.models_dir / "checkpoints").mkdir(exist_ok=True)

        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

    def fetch_repositories(
        self,
        batch_size: int = 100,
        max_repos: Optional[int] = None
    ) -> List[Dict]:
        """Fetch all repositories from API."""
        logger.info("=" * 60)
        logger.info("FETCHING DATA FROM SERVER")
        logger.info("=" * 60)

        # Get total count
        logger.info("Getting total repository count...")
        total_repos = self._get_total_count()
        logger.info(f"Total repositories available: {total_repos}")

        if max_repos:
            total_repos = min(total_repos, max_repos)
            logger.info(f"Limiting to: {max_repos}")

        if total_repos == 0:
            logger.warning("No repositories found on server!")
            return []

        # Fetch in batches
        all_repos = []
        skip = 0

        while skip < total_repos:
            current_batch_size = min(batch_size, total_repos - skip)
            logger.info(f"Fetching batch {skip//batch_size + 1} (repos {skip+1}-{skip+current_batch_size})...")

            batch = self._fetch_batch(skip=skip, limit=current_batch_size)

            if not batch:
                logger.warning("No more data returned, stopping...")
                break

            all_repos.extend(batch)
            logger.info(f"✓ Fetched {len(batch)} repositories (total: {len(all_repos)})")

            skip += batch_size
            time.sleep(0.1)  # Be nice to the server

        logger.info(f"✓ Total fetched: {len(all_repos)} repositories")
        return all_repos

    def _stable_id(self, doc: Dict) -> str:
        """Compute a small stable id for a repository document.

        Uses existing identifiers where possible (`_id`, `id`, `full_name`),
        falls back to owner/name and finally a content hash.
        """
        if not doc:
            return ""

        if doc.get("_id"):
            return str(doc["_id"])

        for key in ("nameWithOwner", "full_name", "repo_id", "id"):
            if doc.get(key):
                return str(doc[key])

        owner = doc.get("owner") or doc.get("owner_login")
        name = doc.get("name")
        if owner and name:
            return f"{owner}/{name}"

        # Fallback: deterministic hash of the doc
        payload = json.dumps(doc, sort_keys=True, default=str).encode("utf-8")
        return hashlib.md5(payload).hexdigest()

    def _dedupe_repositories(self, repositories: List[Dict]) -> List[Dict]:
        """Remove duplicates by stable id, preserving first occurrence."""
        seen = set()
        unique = []
        for r in repositories:
            sid = self._stable_id(r)
            if not sid:
                # keep items without an id
                unique.append(r)
                continue
            if sid in seen:
                continue
            seen.add(sid)
            unique.append(r)
        return unique

    def _get_total_count(self) -> int:
        """Get total number of repositories."""
        try:
            response = requests.post(
                f"{self.api_base_url}/api/mongodb/query",
                headers=self.headers,
                json={
                    "database": "gitquery",
                    "collection": "raw_repositories",
                    "filter": {},
                    "limit": 1,
                    "skip": 0
                },
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            count = result.get("count", 0)

            # If count seems wrong (e.g., 1 when there are clearly more),
            # try to estimate by fetching a larger batch
            if count <= 1:
                logger.warning(f"API reports only {count} repos - attempting to verify...")
                test_response = requests.post(
                    f"{self.api_base_url}/api/mongodb/query",
                    headers=self.headers,
                    json={
                        "database": "gitquery",
                        "collection": "raw_repositories",
                        "filter": {},
                        "limit": 1000,  # Fetch up to 1000 to check
                        "skip": 0
                    },
                    timeout=30
                )
                test_response.raise_for_status()
                test_result = test_response.json()
                actual_docs = len(test_result.get("documents", []))
                if actual_docs > count:
                    logger.info(f"Found {actual_docs} repos (count API was wrong)")
                    # Return a large number to force fetching all
                    return 999999

            return count
        except Exception as e:
            logger.error(f"Error getting count: {e}")
            # Return large number to try fetching anyway
            return 999999

    def _fetch_batch(
        self,
        skip: int,
        limit: int,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """Fetch a single batch of repositories."""
        payload = {
            "database": "gitquery",
            "collection": "raw_repositories",
            "filter": filters or {},
            "limit": limit,
            "skip": skip,
            "sort": {"stars": -1}
        }

        try:
            response = requests.post(
                f"{self.api_base_url}/api/mongodb/query",
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()

            result = response.json()
            return result.get("documents", [])

        except Exception as e:
            logger.error(f"Error fetching batch (skip={skip}): {e}")
            return []

    def check_for_new_data(self, repositories: List[Dict]) -> Tuple[bool, List[Dict]]:
        """Check if there's new data compared to last training run."""
        metadata_path = self.models_dir / "metadata" / "training_metadata_latest.json"

        if not metadata_path.exists():
            logger.info("No previous training found - this is the first run")
            return True, repositories

        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            previous_count = metadata.get("num_repos", 0)
            current_count = len(repositories)

            if current_count > previous_count:
                new_count = current_count - previous_count
                logger.info(f"Found {new_count} new repositories since last training")
                return True, repositories
            else:
                logger.info("No new repositories found")
                return False, []

        except Exception as e:
            logger.warning(f"Could not read previous metadata: {e}")
            return True, repositories

    def save_data_cache(self, repositories: List[Dict]):
        """Save fetched data to cache."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_path = self.data_cache_dir / f"repositories_{timestamp}.json"

        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(repositories, f, indent=2, default=str)

        logger.info(f"✓ Cached {len(repositories)} repositories to {cache_path}")

        # Save as latest
        latest_path = self.data_cache_dir / "repositories_latest.json"
        with open(latest_path, 'w', encoding='utf-8') as f:
            json.dump(repositories, f, indent=2, default=str)

    def prepare_repo_text(self, repo: Dict) -> str:
        """Prepare repository text for embedding."""
        from .utils import prepare_repo_text as _prepare_repo_text
        return _prepare_repo_text(repo)

    def train_embeddings(
        self,
        repositories: List[Dict],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32
    ) -> Tuple[np.ndarray, List[str], Dict]:
        """Train/generate embeddings for repositories."""
        logger.info("=" * 60)
        logger.info("TRAINING EMBEDDINGS")
        logger.info("=" * 60)
        logger.info(f"Model: {model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Repositories: {len(repositories)}")
        logger.info("=" * 60)

        # Load model
        logger.info("Loading embedding model...")
        model = SentenceTransformer(model_name, device=self.device)
        logger.info("✓ Model loaded")

        # Prepare texts
        logger.info("Preparing repository texts...")
        texts = []
        repo_ids = []

        for repo in repositories:
            text = self.prepare_repo_text(repo)
            texts.append(text)
            repo_id = self._stable_id(repo)
            repo_ids.append(repo_id)

        logger.info(f"✓ Prepared {len(texts)} texts")

        # Generate embeddings
        logger.info("Generating embeddings...")
        start_time = time.time()

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Important for cosine similarity
        )

        elapsed = time.time() - start_time
        logger.info(f"✓ Generated {len(embeddings)} embeddings in {elapsed:.2f}s")
        logger.info(f"  Embedding dimension: {embeddings.shape[1]}")
        logger.info(f"  Speed: {len(embeddings)/elapsed:.2f} repos/sec")

        # Create metadata
        metadata = {
            "model_name": model_name,
            "num_repos": len(repositories),
            "embedding_dim": int(embeddings.shape[1]),
            "device": self.device,
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "batch_size": batch_size,
            "training_time_seconds": elapsed,
            "normalized": True
        }

        return embeddings, repo_ids, metadata

    def save_model(
        self,
        embeddings: np.ndarray,
        repo_ids: List[str],
        metadata: Dict
    ):
        """Save trained model to disk."""
        logger.info("=" * 60)
        logger.info("SAVING MODEL")
        logger.info("=" * 60)

        timestamp = metadata["timestamp"]

        # Save embeddings
        embeddings_path = self.models_dir / "vectors" / f"repo_embeddings_{timestamp}.npy"
        np.save(embeddings_path, embeddings)
        logger.info(f"✓ Saved embeddings: {embeddings_path}")

        # Save as latest
        latest_path = self.models_dir / "vectors" / "repo_embeddings_latest.npy"
        np.save(latest_path, embeddings)
        logger.info(f"✓ Saved latest: {latest_path}")

        # Save mapping (repo_id -> index)
        mapping = {repo_id: idx for idx, repo_id in enumerate(repo_ids)}
        mapping_path = self.models_dir / "metadata" / f"repo_mapping_{timestamp}.json"
        with open(mapping_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        logger.info(f"✓ Saved mapping: {mapping_path}")

        # Save latest mapping
        latest_mapping_path = self.models_dir / "metadata" / "repo_mapping_latest.json"
        with open(latest_mapping_path, 'w') as f:
            json.dump(mapping, f, indent=2)

        # Save metadata
        metadata_path = self.models_dir / "metadata" / f"training_metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"✓ Saved metadata: {metadata_path}")

        # Save latest metadata
        latest_metadata_path = self.models_dir / "metadata" / "training_metadata_latest.json"
        with open(latest_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info("=" * 60)
        logger.info("✓ MODEL SAVED SUCCESSFULLY")
        logger.info("=" * 60)

    def register_model(self, metadata: Dict, repo_ids: List[str]):
        """Save model registration for the recommender service to pick up.

        Writes model_registry_latest.json to models/metadata/ with
        ModelMetadata-compatible fields. The recommender service reads this
        file as an alternative to a MongoDB lookup, since the training
        container does not set up a Motor async connection.
        """
        registration = {
            "model_id": f"embedding-{metadata['timestamp']}",
            "model_type": "embedding",
            "model_name": metadata["model_name"],
            "version": metadata["timestamp"],
            "path": "vectors/repo_embeddings_latest.npy",
            "mapping_path": "metadata/repo_mapping_latest.json",
            "is_active": True,
            "variant": "default",
            "num_repos": metadata["num_repos"],
            "embedding_dim": metadata["embedding_dim"],
            "trained_at": metadata["timestamp"],
            "metrics": {
                "training_time_seconds": float(metadata["training_time_seconds"]),
                "num_repos": float(metadata["num_repos"]),
                "embedding_dim": float(metadata["embedding_dim"]),
            },
            "hyperparameters": {
                "batch_size": metadata["batch_size"],
                "device": metadata["device"],
                "normalized": metadata.get("normalized", True),
            },
        }

        reg_path = self.models_dir / "metadata" / "model_registry_latest.json"
        with open(reg_path, 'w') as f:
            json.dump(registration, f, indent=2)
        logger.info(f"✓ Model registered: {reg_path}")

    def upload_to_qdrant(self, embeddings: np.ndarray, repo_ids: List[str], metadata: Dict):
        """Upload trained embeddings to Qdrant vector store."""
        logger.info("=" * 60)
        logger.info("UPLOADING TO QDRANT")
        logger.info("=" * 60)

        qdrant_api_key = os.getenv("APIKEY_QDRANT", self.api_key)
        collection = os.getenv("QDRANT_COLLECTION", "repositories_embeddings")
        batch_size = int(os.getenv("UPLOAD_BATCH_SIZE", "100"))

        uploader = EmbeddingUploader(
            base_url=self.api_base_url,
            qdrant_api_key=qdrant_api_key,
            models_dir=str(self.models_dir)
        )

        uploaded = uploader.upload_all(
            collection=collection,
            batch_size=batch_size
        )

        logger.info(f"Qdrant upload finished — {uploaded} points uploaded")

    def print_summary(self, repositories: List[Dict], metadata: Dict):
        """Print training summary."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 60)

        logger.info(f"\nDataset:")
        logger.info(f"  Total repositories: {len(repositories)}")

        # Language distribution
        languages = {}
        for repo in repositories:
            lang = repo.get("language") or "Unknown"
            languages[lang] = languages.get(lang, 0) + 1

        logger.info(f"\nTop 10 languages:")
        for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True)[:10]:
            pct = (count / len(repositories)) * 100
            logger.info(f"  {lang}: {count} ({pct:.1f}%)")

        # Star statistics
        stars = [repo.get("stargazers_count", 0) or repo.get("stars", 0) for repo in repositories]
        if stars:
            logger.info(f"\nStar statistics:")
            logger.info(f"  Min: {min(stars):,}")
            logger.info(f"  Max: {max(stars):,}")
            logger.info(f"  Avg: {sum(stars)/len(stars):,.0f}")

        logger.info(f"\nModel:")
        logger.info(f"  Model name: {metadata['model_name']}")
        logger.info(f"  Embedding dimension: {metadata['embedding_dim']}")
        logger.info(f"  Device: {metadata['device']}")
        logger.info(f"  Training time: {metadata['training_time_seconds']:.2f}s")

        logger.info(f"\nOutput:")
        logger.info(f"  Models directory: {self.models_dir}")
        logger.info(f"  Data cache: {self.data_cache_dir}")

        logger.info("")
        logger.info("=" * 60)

    def run(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        fetch_batch_size: int = 100,
        max_repos: Optional[int] = None,
        skip_if_no_new_data: bool = True
    ):
        """Run the full pipeline: fetch + train."""
        logger.info("")
        logger.info("=" * 60)
        logger.info("UNIFIED TRAINING PIPELINE")
        logger.info("=" * 60)
        logger.info(f"API: {self.api_base_url}")
        logger.info(f"Model: {model_name}")
        logger.info(f"Max repos: {max_repos or 'all'}")
        logger.info("=" * 60)
        logger.info("")

        start_time = time.time()

        try:
            # Step 1: Fetch data
            repositories = self.fetch_repositories(
                batch_size=fetch_batch_size,
                max_repos=max_repos
            )

            if not repositories:
                logger.error("❌ No repositories fetched - cannot train!")
                return

            # Deduplicate fetched repositories (avoid page/pull duplicates)
            before_count = len(repositories)
            repositories = self._dedupe_repositories(repositories)
            removed = before_count - len(repositories)
            if removed:
                logger.info(f"Removed {removed} duplicate repositories before training")

            # Step 2: Check for new data
            if skip_if_no_new_data:
                has_new_data, _ = self.check_for_new_data(repositories)
                if not has_new_data:
                    logger.info("⏭ Skipping training - no new data")
                    return

            # Step 3: Cache data
            self.save_data_cache(repositories)

            # Step 4: Train embeddings
            embeddings, repo_ids, metadata = self.train_embeddings(
                repositories=repositories,
                model_name=model_name,
                batch_size=batch_size
            )

            # Step 5: Save model
            self.save_model(embeddings, repo_ids, metadata)

            # Step 6: Register model so recommender service can discover it
            self.register_model(metadata, repo_ids)

            # Step 7: Upload embeddings to Qdrant
            skip_qdrant = os.getenv("SKIP_QDRANT_UPLOAD", "false").lower() == "true"
            if skip_qdrant:
                logger.info("Skipping Qdrant upload (SKIP_QDRANT_UPLOAD=true)")
            else:
                self.upload_to_qdrant(embeddings, repo_ids, metadata)

            # Step 8: Print summary
            self.print_summary(repositories, metadata)

            total_time = time.time() - start_time
            logger.info("")
            logger.info("=" * 60)
            logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info(f"Total time: {total_time/60:.2f} minutes")
            logger.info(f"Repositories processed: {len(repositories)}")
            logger.info(f"Model saved to: {self.models_dir}")
            logger.info("=" * 60)
            logger.info("")

        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
            raise


def main():
    """Main entry point for containerized training."""
    # Read from environment variables
    api_base_url = os.getenv("API_BASE_URL")
    api_key = os.getenv("APIKEY_MONGODB")

    if not api_base_url or not api_key:
        raise ValueError(
            "Missing required environment variables!\n"
            "Please set API_BASE_URL and APIKEY_MONGODB in your .env file"
        )

    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    batch_size = int(os.getenv("BATCH_SIZE", "32"))
    fetch_batch_size = int(os.getenv("FETCH_BATCH_SIZE", "100"))
    max_repos = os.getenv("MAX_REPOS")
    max_repos = int(max_repos) if max_repos else None
    skip_if_no_new_data = os.getenv("SKIP_IF_NO_NEW_DATA", "true").lower() == "true"

    # Initialize pipeline
    pipeline = UnifiedTrainingPipeline(
        api_base_url=api_base_url,
        api_key=api_key
    )

    # Run pipeline
    pipeline.run(
        model_name=model_name,
        batch_size=batch_size,
        fetch_batch_size=fetch_batch_size,
        max_repos=max_repos,
        skip_if_no_new_data=skip_if_no_new_data
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n⚠ Training interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}", exc_info=True)
        exit(1)
