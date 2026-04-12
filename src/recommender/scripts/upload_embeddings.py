"""Upload locally trained embeddings to Qdrant via server API."""

import json
import os
import numpy as np
import requests
from pathlib import Path
from typing import List, Dict
import time
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EmbeddingUploader:
    """Upload embeddings to Qdrant through the server API."""

    def __init__(self, base_url: str, qdrant_api_key: str, models_dir: str = "./models"):
        self.base_url = base_url.rstrip("/")
        self.models_dir = Path(models_dir)

        self.headers = {"Authorization": f"Bearer {qdrant_api_key}", "Content-Type": "application/json"}

    def load_embeddings(self) -> tuple:
        """Load latest embeddings and mapping."""

        # Load embeddings
        embeddings_path = self.models_dir / "vectors" / "repo_embeddings_latest.npy"
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")

        embeddings = np.load(embeddings_path)
        logger.info(f"Loaded embeddings: {embeddings.shape}")

        # Load mapping
        mapping_path = self.models_dir / "metadata" / "repo_mapping_latest.json"
        if not mapping_path.exists():
            raise FileNotFoundError(f"Mapping not found: {mapping_path}")

        with open(mapping_path, "r") as f:
            mapping = json.load(f)

        # Support mapping as legacy dict {repo_id: index} or new list format
        if isinstance(mapping, dict):
            # convert to list format
            mapping_list = [{"repo_id": k, "index": int(v), "hash": "", "full_name": None} for k, v in mapping.items()]
        elif isinstance(mapping, list):
            mapping_list = mapping
        else:
            raise ValueError("Unknown mapping format")

        logger.info(f"Loaded mapping: {len(mapping_list)} repositories")

        # Load metadata
        metadata_path = self.models_dir / "metadata" / "training_metadata_latest.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            logger.info("Loaded metadata")

        return embeddings, mapping_list, metadata

    def ensure_collection(self, collection: str, vector_size: int = 384):
        """Create the Qdrant collection if it does not already exist."""
        payload = {"vectors": {"size": vector_size, "distance": "Cosine"}}

        try:
            response = requests.put(
                f"{self.base_url}/api/qdrant/collections/{collection}", headers=self.headers, json=payload
            )
            response.raise_for_status()
            logger.info(f"Created Qdrant collection '{collection}' (size={vector_size})")
        except requests.exceptions.HTTPError:
            # Collection already exists — that is fine
            logger.info(f"Qdrant collection '{collection}' already exists (or creation was a no-op)")
        except Exception as e:
            logger.warning(f"Could not ensure Qdrant collection exists: {e}")

    def upload_batch(self, collection: str, points: List[Dict], wait: bool = True) -> Dict:
        """Upload a batch of points to Qdrant."""

        payload = {"points": points, "wait": wait}

        try:
            response = requests.post(
                f"{self.base_url}/api/qdrant/collections/{collection}/points", headers=self.headers, json=payload
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error uploading batch: {e}")
            return {"status": "error", "error": str(e)}

    def upload_all(self, collection: str = "repositories_embeddings", batch_size: int = 100):
        """Upload all embeddings to Qdrant."""

        logger.info("=" * 60)
        logger.info(f"Uploading embeddings to Qdrant collection: {collection}")
        logger.info("=" * 60)

        # Load data
        embeddings, mapping, metadata = self.load_embeddings()

        # Ensure collection exists before uploading
        vector_size = embeddings.shape[1] if len(embeddings.shape) > 1 else 384
        self.ensure_collection(collection, vector_size=vector_size)

        # Prepare points from mapping list
        logger.info(f"Preparing {len(mapping)} points...")

        # Deduplicate mapping by repo_id (preserve first occurrence)
        seen = set()
        deduped = []
        for m in mapping:
            rid = m.get("repo_id")
            if not rid:
                logger.warning("Skipping mapping entry with empty repo_id")
                continue
            if rid in seen:
                logger.warning(f"Duplicate mapping entry for repo_id {rid} - skipping subsequent occurrence")
                continue
            seen.add(rid)
            deduped.append(m)

        total_points = len(deduped)
        uploaded = 0

        # Upload in batches
        for i in range(0, total_points, batch_size):
            batch = deduped[i : i + batch_size]
            batch_points = []

            for entry in batch:
                repo_id = entry.get("repo_id")
                idx = entry.get("index")
                hsh = entry.get("hash")

                if idx is None:
                    logger.warning(f"No index for repo {repo_id}, skipping")
                    continue

                try:
                    idx_int = int(idx)
                except (TypeError, ValueError):
                    logger.warning(f"Invalid index for repo {repo_id}: {idx}, skipping")
                    continue

                if idx_int < 0 or idx_int >= embeddings.shape[0]:
                    logger.warning(f"Index out of range for repo {repo_id}: {idx_int}, skipping")
                    continue

                vector = embeddings[idx_int].tolist()
                point = {
                    "id": repo_id,
                    "vector": vector,
                    "payload": {
                        "repo_id": repo_id,
                        "model": metadata.get("model_name", "unknown"),
                        "timestamp": metadata.get("timestamp", "unknown"),
                        "hash": hsh,
                    },
                }
                batch_points.append(point)

            logger.info(
                f"Uploading batch {i // batch_size + 1}/"
                f"{(total_points + batch_size - 1) // batch_size} "
                f"({len(batch_points)} points)..."
            )

            result = self.upload_batch(collection, batch_points)

            if result.get("status") != "error":
                uploaded += len(batch_points)
                logger.info(f"Uploaded {uploaded}/{total_points} points")
            else:
                logger.error(f"Failed to upload batch: {result.get('error')}")

            # Rate limiting
            time.sleep(0.5)

        logger.info("=" * 60)
        logger.info("Upload complete!")
        logger.info(f"  Total uploaded: {uploaded}/{total_points}")
        logger.info("=" * 60)

        return uploaded


def main():
    """Main upload function — reads config from environment variables."""

    logger.info("=" * 60)
    logger.info("UPLOAD EMBEDDINGS TO QDRANT")
    logger.info("=" * 60)

    # Read configuration from environment variables
    base_url = os.getenv("API_BASE_URL")
    qdrant_api_key = os.getenv("APIKEY_QDRANT")

    if not base_url or not qdrant_api_key:
        raise ValueError("Missing required environment variables!\nPlease set API_BASE_URL and APIKEY_QDRANT.")

    collection = os.getenv("QDRANT_COLLECTION", "repositories_embeddings")
    batch_size = int(os.getenv("UPLOAD_BATCH_SIZE", "100"))
    models_dir = os.getenv("MODELS_DIR", "./models")

    # Initialize uploader
    uploader = EmbeddingUploader(base_url=base_url, qdrant_api_key=qdrant_api_key, models_dir=models_dir)

    # Upload
    uploader.upload_all(collection=collection, batch_size=batch_size)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\nUpload interrupted by user")
    except Exception as e:
        logger.error(f"\nUpload failed: {e}")
        import traceback

        traceback.print_exc()
