"""Cross-encoder reranker trainer."""

import logging
from typing import Dict, Any, List
from sentence_transformers import CrossEncoder, InputExample
from datetime import datetime
import os
import shutil
import glob

from ..config import settings
from ..models import ModelMetadata

logger = logging.getLogger(__name__)


class RerankerTrainer:
    """
    Trainer for cross-encoder reranking models.

    Cross-encoders are more accurate but slower than bi-encoders.
    Used for reranking top K candidates.
    """

    def __init__(self, base_model: str = None, checkpoint_save_total_limit: int = 3):
        self.base_model = base_model or settings.cross_encoder_model_name
        self.checkpoint_save_total_limit = checkpoint_save_total_limit
        self.model = None

    async def train(
        self,
        training_data: Dict[str, List],
        variant: str,
        epochs: int = 3,
        batch_size: int = 16,
    ) -> Dict[str, Any]:
        """
        Train/fine-tune cross-encoder model.

        Args:
            training_data: Dictionary with query-repo pairs and labels
            variant: Model variant name
            epochs: Number of training epochs
            batch_size: Training batch size

        Returns:
            Training metrics
        """
        logger.info(f"Training cross-encoder model: {self.base_model}")

        # Load base model
        self.model = CrossEncoder(self.base_model, num_labels=1)

        # Prepare training examples
        train_examples = self._prepare_examples(training_data)

        if not train_examples:
            logger.warning("No training examples available")
            return {}

        # Train
        logger.info(f"Starting training with {len(train_examples)} examples")

        output_path = os.path.join(
            settings.model_path,
            f"reranker_{variant}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        )

        def train_callback(score: float, epoch: int, steps: int):
            """Callback to save checkpoints during training."""
            metrics = {"score": score, "steps": steps}
            # epoch is 0-indexed in callback
            self._save_checkpoint(epoch + 1, metrics, variant)

        self.model.fit(
            train_dataloader=self._create_dataloader(train_examples, batch_size),
            epochs=epochs,
            warmup_steps=100,
            output_path=output_path,
            show_progress_bar=True,
            callback=train_callback
        )

        logger.info(f"Model saved to: {output_path}")

        # Save metadata via registry
        metadata = ModelMetadata(
            model_id=f"reranker_{variant}_{datetime.utcnow().timestamp()}",
            model_type="cross_encoder",
            variant=variant,
            version="1.0.0",
            path=os.path.relpath(output_path, settings.model_path),
            hyperparameters={
                "base_model": self.base_model,
                "epochs": epochs,
                "batch_size": batch_size,
            },
            metrics={"num_examples": float(len(train_examples))},
            trained_at=datetime.utcnow(),
            is_active=False,
            status="candidate"
        )

        from ..services.registry_service import ModelRegistryService
        registry = ModelRegistryService()
        await registry.register_model(metadata)

        return {
            "model_path": output_path,
            "num_examples": len(train_examples),
            "epochs": epochs,
        }

    def _save_checkpoint(self, epoch: int, metrics: Dict[str, Any], variant: str):
        """
        Save training checkpoint and prune old ones.
        
        Args:
            epoch: Current epoch number (1-based)
            metrics: Current metrics
            variant: Model variant name
        """
        checkpoint_dir = os.path.join(settings.checkpoint_path, variant)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        epoch_path = os.path.join(checkpoint_dir, f"epoch_{epoch}")
        
        logger.info(f"Saving checkpoint to {epoch_path}")
        self.model.save(epoch_path)
        
        # Prune old checkpoints
        self._prune_checkpoints(checkpoint_dir)

    def _prune_checkpoints(self, checkpoint_dir: str):
        """
        Keep only the latest N checkpoints in the directory.
        
        Args:
            checkpoint_dir: Directory containing checkpoints
        """
        checkpoints = glob.glob(os.path.join(checkpoint_dir, "epoch_*"))
        
        # Sort by epoch number
        def get_epoch_num(path):
            try:
                dirname = os.path.basename(path)
                return int(dirname.split("_")[-1])
            except (ValueError, IndexError):
                return -1
                
        checkpoints.sort(key=get_epoch_num)
        
        if len(checkpoints) > self.checkpoint_save_total_limit:
            to_delete = checkpoints[:-self.checkpoint_save_total_limit]
            for path in to_delete:
                logger.info(f"Pruning old checkpoint: {path}")
                if os.path.isdir(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)

    def _prepare_examples(self, training_data: Dict[str, List]) -> List[InputExample]:
        """
        Prepare training examples for cross-encoder.

        Creates (query, repo, label) tuples.
        Label: 1 for positive (clicked/saved), 0 for negative.
        """
        examples = []

        # Positive examples
        queries = training_data.get("queries", [])
        positive_repos = training_data.get("positive_repos", [])

        for query, repo in zip(queries, positive_repos):
            repo_text = self._repo_to_text(repo)
            example = InputExample(texts=[query, repo_text], label=1.0)
            examples.append(example)

        # Negative examples
        negative_repos = training_data.get("negative_repos", [])

        for query, repo in zip(queries, negative_repos):
            repo_text = self._repo_to_text(repo)
            example = InputExample(texts=[query, repo_text], label=0.0)
            examples.append(example)

        return examples

    def _repo_to_text(self, repo: Dict[str, Any]) -> str:
        """Convert repository data to text."""
        parts = []

        if repo.get("name"):
            parts.append(repo["name"])

        if repo.get("description"):
            parts.append(repo["description"])

        if repo.get("language"):
            parts.append(f"Language: {repo['language']}")

        return " | ".join(parts)

    def _create_dataloader(self, examples: List[InputExample], batch_size: int):
        """Create DataLoader for training."""
        from torch.utils.data import DataLoader
        return DataLoader(examples, shuffle=True, batch_size=batch_size)

