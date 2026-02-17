"""Upload locally trained embeddings to Qdrant via server API."""

import json
import numpy as np
import requests
from pathlib import Path
from typing import List, Dict
import time


class EmbeddingUploader:
    """Upload embeddings to Qdrant through the server API."""

    def __init__(
        self,
        base_url: str,
        qdrant_api_key: str,
        models_dir: str = "./models"
    ):
        self.base_url = base_url.rstrip('/')
        self.models_dir = Path(models_dir)

        self.headers = {
            "Authorization": f"Bearer {qdrant_api_key}",
            "Content-Type": "application/json"
        }

    def load_embeddings(self) -> tuple:
        """Load latest embeddings and mapping."""

        # Load embeddings
        embeddings_path = self.models_dir / "vectors" / "repo_embeddings_latest.npy"
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")

        embeddings = np.load(embeddings_path)
        print(f"✓ Loaded embeddings: {embeddings.shape}")

        # Load mapping
        mapping_path = self.models_dir / "metadata" / "repo_mapping_latest.json"
        if not mapping_path.exists():
            raise FileNotFoundError(f"Mapping not found: {mapping_path}")

        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
        print(f"✓ Loaded mapping: {len(mapping)} repositories")

        # Load metadata
        metadata_path = self.models_dir / "metadata" / "training_metadata_latest.json"
        metadata = {}
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"✓ Loaded metadata")

        return embeddings, mapping, metadata

    def upload_batch(
        self,
        collection: str,
        points: List[Dict],
        wait: bool = True
    ) -> Dict:
        """Upload a batch of points to Qdrant."""

        payload = {
            "points": points,
            "wait": wait
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/qdrant/collections/{collection}/points",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error uploading batch: {e}")
            return {"status": "error", "error": str(e)}

    def upload_all(
        self,
        collection: str = "repositories_embeddings",
        batch_size: int = 100
    ):
        """Upload all embeddings to Qdrant."""

        print("\n" + "="*60)
        print(f"Uploading embeddings to Qdrant collection: {collection}")
        print("="*60)

        # Load data
        embeddings, mapping, metadata = self.load_embeddings()

        # Prepare points
        print(f"\nPreparing {len(mapping)} points...")

        repo_ids = list(mapping.keys())
        total_points = len(repo_ids)
        uploaded = 0

        # Upload in batches
        for i in range(0, total_points, batch_size):
            batch_ids = repo_ids[i:i+batch_size]
            batch_points = []

            for repo_id in batch_ids:
                idx = mapping[repo_id]
                vector = embeddings[idx].tolist()

                point = {
                    "id": repo_id,
                    "vector": vector,
                    "payload": {
                        "repo_id": repo_id,
                        "model": metadata.get("model_name", "unknown"),
                        "timestamp": metadata.get("timestamp", "unknown")
                    }
                }
                batch_points.append(point)

            print(f"Uploading batch {i//batch_size + 1}/{(total_points + batch_size - 1)//batch_size} ({len(batch_points)} points)...")

            result = self.upload_batch(collection, batch_points)

            if result.get("status") != "error":
                uploaded += len(batch_points)
                print(f"✓ Uploaded {uploaded}/{total_points} points")
            else:
                print(f"❌ Failed to upload batch: {result.get('error')}")

            # Rate limiting
            time.sleep(0.5)

        print("\n" + "="*60)
        print(f"✓ Upload complete!")
        print(f"  Total uploaded: {uploaded}/{total_points}")
        print("="*60)


def main():
    """Main upload function."""

    print("\n" + "="*60)
    print("UPLOAD EMBEDDINGS TO QDRANT")
    print("="*60)

    # Get configuration
    BASE_URL = input("Enter server URL (e.g., http://your-server.com or http://localhost:8000): ").strip()
    QDRANT_API_KEY = input("Enter Qdrant API key: ").strip()
    COLLECTION = input("Collection name (default: repositories_embeddings): ").strip() or "repositories_embeddings"
    BATCH_SIZE = int(input("Batch size (default: 100): ").strip() or "100")

    # Initialize uploader
    uploader = EmbeddingUploader(
        base_url=BASE_URL,
        qdrant_api_key=QDRANT_API_KEY
    )

    # Confirm
    proceed = input("\nProceed with upload? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Upload cancelled.")
        return

    # Upload
    uploader.upload_all(
        collection=COLLECTION,
        batch_size=BATCH_SIZE
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nUpload interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Upload failed: {e}")
        import traceback
        traceback.print_exc()

