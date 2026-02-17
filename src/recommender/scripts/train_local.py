"""Train ML models locally using fetched data with checkpointing support."""

import json
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from sentence_transformers import SentenceTransformer
import numpy as np


class LocalModelTrainer:
    """Train and save ML models locally with checkpointing."""

    def __init__(
        self,
        data_dir: str = "./data/training",
        models_dir: str = "src/recommender/models"
    ):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.models_dir / "embeddings").mkdir(exist_ok=True)
        (self.models_dir / "vectors").mkdir(exist_ok=True)
        (self.models_dir / "metadata").mkdir(exist_ok=True)
        (self.models_dir / "checkpoints").mkdir(exist_ok=True)

    def save_checkpoint(
        self,
        embeddings: np.ndarray,
        repo_ids: List[str],
        processed_count: int,
        total_count: int,
        checkpoint_name: str = "checkpoint"
    ):
        """Save training checkpoint."""
        checkpoint_dir = self.models_dir / "checkpoints"
        checkpoint_path = checkpoint_dir / f"{checkpoint_name}.npz"

        np.savez(
            checkpoint_path,
            embeddings=embeddings,
            repo_ids=repo_ids,
            processed_count=processed_count,
            total_count=total_count
        )

        print(f"✓ Checkpoint saved: {checkpoint_path} ({processed_count}/{total_count})")

    def load_checkpoint(self, checkpoint_name: str = "checkpoint") -> Optional[Tuple]:
        """Load training checkpoint if exists."""
        checkpoint_path = self.models_dir / "checkpoints" / f"{checkpoint_name}.npz"

        if not checkpoint_path.exists():
            return None

        try:
            data = np.load(checkpoint_path, allow_pickle=True)
            embeddings = data['embeddings']
            repo_ids = data['repo_ids'].tolist()
            processed_count = int(data['processed_count'])
            total_count = int(data['total_count'])

            print(f"✓ Loaded checkpoint: {processed_count}/{total_count} processed")
            return embeddings, repo_ids, processed_count
        except Exception as e:
            print(f"⚠ Failed to load checkpoint: {e}")
            return None

    def load_repositories(self, filename: str = "repositories_latest.json") -> List[Dict]:
        """Load repository data from JSON."""
        filepath = self.data_dir / filename
        if not filepath.exists():
            print(f"❌ Repository file not found: {filepath}")
            return []

        with open(filepath, 'r', encoding='utf-8') as f:
            repos = json.load(f)

        print(f"✓ Loaded {len(repos)} repositories from {filepath}")
        return repos

    def prepare_repo_text(self, repo: Dict) -> str:
        """Prepare repository text for embedding."""
        parts = []

        # Repository name
        if repo.get("name"):
            parts.append(f"Repository: {repo['name']}")

        # Description
        if repo.get("description"):
            parts.append(f"Description: {repo['description']}")

        # Topics/tags
        if repo.get("topics"):
            topics = repo["topics"]
            if isinstance(topics, list):
                parts.append(f"Topics: {', '.join(topics)}")

        # Language
        if repo.get("language"):
            parts.append(f"Language: {repo['language']}")

        # README (truncated)
        if repo.get("readme"):
            readme = repo["readme"][:500]  # First 500 chars
            parts.append(f"README: {readme}")

        return " | ".join(parts)

    def generate_embeddings(
        self,
        repositories: List[Dict],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        resume_from_checkpoint: bool = True,
        save_checkpoints: bool = True,
        checkpoint_interval: int = 100
    ):
        """Generate embeddings for repositories with checkpointing."""

        print("\n" + "="*60)
        print(f"Generating embeddings using {model_name}")
        print("="*60)

        # Try to resume from checkpoint
        start_idx = 0
        all_embeddings = []
        all_repo_ids = []

        if resume_from_checkpoint:
            checkpoint_data = self.load_checkpoint()
            if checkpoint_data:
                all_embeddings, all_repo_ids, start_idx = checkpoint_data
                all_embeddings = all_embeddings.tolist()
                print(f"✓ Resuming from checkpoint at index {start_idx}")

        # Load model
        print(f"\nLoading embedding model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        model = SentenceTransformer(model_name, device=device)

        # Prepare texts for remaining repositories
        print(f"\nPreparing repository texts...")
        remaining_repos = repositories[start_idx:]

        print(f"✓ Processing {len(remaining_repos)} repositories (starting from {start_idx})")

        # Process in batches with checkpointing
        for i in range(0, len(remaining_repos), batch_size):
            batch_repos = remaining_repos[i:i + batch_size]

            # Prepare batch
            texts = [self.prepare_repo_text(repo) for repo in batch_repos]
            repo_ids = [str(repo.get("_id", repo.get("id", ""))) for repo in batch_repos]

            # Generate embeddings for batch
            batch_embeddings = model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True
            )

            # Accumulate results
            all_embeddings.extend(batch_embeddings.tolist())
            all_repo_ids.extend(repo_ids)

            current_count = start_idx + i + len(batch_repos)
            print(f"Progress: {current_count}/{len(repositories)} ({current_count/len(repositories)*100:.1f}%)")

            # Save checkpoint periodically
            if save_checkpoints and (i + batch_size) % checkpoint_interval == 0:
                embeddings_array = np.array(all_embeddings)
                self.save_checkpoint(
                    embeddings_array,
                    all_repo_ids,
                    current_count,
                    len(repositories)
                )

        # Convert final embeddings to numpy array
        embeddings = np.array(all_embeddings)

        print(f"\n✓ Generated {len(embeddings)} embeddings")
        print(f"  Embedding dimension: {embeddings.shape[1]}")

        # Save embeddings
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save as numpy array
        embeddings_path = self.models_dir / "vectors" / f"repo_embeddings_{timestamp}.npy"
        np.save(embeddings_path, embeddings)
        print(f"✓ Saved embeddings to {embeddings_path}")

        # Save latest
        latest_path = self.models_dir / "vectors" / "repo_embeddings_latest.npy"
        np.save(latest_path, embeddings)
        print(f"✓ Saved latest embeddings to {latest_path}")

        # Save mapping (repo_id -> index)
        mapping = {repo_id: idx for idx, repo_id in enumerate(all_repo_ids)}
        mapping_path = self.models_dir / "metadata" / f"repo_mapping_{timestamp}.json"
        with open(mapping_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        print(f"✓ Saved mapping to {mapping_path}")

        # Save latest mapping
        latest_mapping_path = self.models_dir / "metadata" / "repo_mapping_latest.json"
        with open(latest_mapping_path, 'w') as f:
            json.dump(mapping, f, indent=2)

        # Save metadata
        metadata = {
            "model_name": model_name,
            "num_repos": len(repositories),
            "embedding_dim": int(embeddings.shape[1]),
            "device": device,
            "timestamp": timestamp,
            "batch_size": batch_size
        }

        metadata_path = self.models_dir / "metadata" / f"training_metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata to {metadata_path}")

        # Save latest metadata
        latest_metadata_path = self.models_dir / "metadata" / "training_metadata_latest.json"
        with open(latest_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Clean up checkpoint
        if save_checkpoints:
            checkpoint_path = self.models_dir / "checkpoints" / "checkpoint.npz"
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                print(f"✓ Cleaned up checkpoint file")

        return embeddings, all_repo_ids, metadata

    def create_summary(self, repositories: List[Dict], metadata: Dict):
        """Create training summary."""

        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)

        print(f"\nDataset:")
        print(f"  Total repositories: {len(repositories)}")

        # Language distribution
        languages = {}
        for repo in repositories:
            lang = repo.get("language", "Unknown")
            languages[lang] = languages.get(lang, 0) + 1

        print(f"\nTop languages:")
        for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {lang}: {count}")

        print(f"\nModel:")
        print(f"  Model name: {metadata['model_name']}")
        print(f"  Embedding dimension: {metadata['embedding_dim']}")
        print(f"  Device: {metadata['device']}")
        print(f"  Batch size: {metadata['batch_size']}")

        print(f"\nOutput:")
        print(f"  Models directory: {self.models_dir}")
        print(f"  Timestamp: {metadata['timestamp']}")

        print("\n" + "="*60)
        print("✓ Training complete!")
        print("="*60)

        print("\nNext steps:")
        print("  1. Upload embeddings to Qdrant (optional)")
        print("  2. Start recommender service: python -m src.recommender")
        print("  3. Test recommendations locally")
        print("="*60)


def main():
    """Main training function."""

    print("\n" + "="*60)
    print("LOCAL MODEL TRAINING")
    print("="*60)

    # Initialize trainer
    trainer = LocalModelTrainer()

    # Load data
    print("\nLoading repository data...")
    repositories = trainer.load_repositories()

    if not repositories:
        print("\n❌ No repository data found!")
        print("\nPlease run: python -m src.recommender.scripts.fetch_data_from_server")
        return

    # Ask for confirmation
    print(f"\nReady to train on {len(repositories)} repositories")
    print(f"Models will be saved to: {trainer.models_dir}")

    model_name = input("\nEmbedding model (default: sentence-transformers/all-MiniLM-L6-v2): ").strip()
    if not model_name:
        model_name = "sentence-transformers/all-MiniLM-L6-v2"

    batch_size = input("Batch size (default: 32): ").strip()
    batch_size = int(batch_size) if batch_size else 32

    resume = input("Resume from checkpoint if exists? (y/n, default: y): ").strip().lower()
    resume_from_checkpoint = resume != 'n'

    proceed = input("\nProceed with training? (y/n): ").strip().lower()
    if proceed != 'y':
        print("Training cancelled.")
        return

    # Generate embeddings
    embeddings, repo_ids, metadata = trainer.generate_embeddings(
        repositories,
        model_name=model_name,
        batch_size=batch_size,
        resume_from_checkpoint=resume_from_checkpoint,
        save_checkpoints=True,
        checkpoint_interval=100
    )

    # Create summary
    trainer.create_summary(repositories, metadata)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        print("   Checkpoint saved - you can resume later with the same command")
    except Exception as e:
        print(f"\n\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
