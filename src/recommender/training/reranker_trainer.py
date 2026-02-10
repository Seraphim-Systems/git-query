"""Cross-encoder reranker trainer."""

import logging
from typing import Dict, Any, List
from sentence_transformers import CrossEncoder, InputExample
from datetime import datetime
import os

from ..config import settings
from ..models import ModelMetadata

logger = logging.getLogger(__name__)


class RerankerTrainer:
    """
    Trainer for cross-encoder reranking models.

    Cross-encoders are more accurate but slower than bi-encoders.
    Used for reranking top K candidates.
    """

    def __init__(self, base_model: str = None):
        self.base_model = base_model or settings.cross_encoder_model_name
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

        self.model.fit(
            train_dataloader=self._create_dataloader(train_examples, batch_size),
            epochs=epochs,
            warmup_steps=100,
            output_path=output_path,
            show_progress_bar=True,
        )

        logger.info(f"Model saved to: {output_path}")

        # Save metadata
        metadata = ModelMetadata(
            model_id=f"reranker_{variant}_{datetime.utcnow().timestamp()}",
            model_type="cross_encoder",
            variant=variant,
            version="1.0.0",
            model_path=output_path,
            hyperparameters={
                "base_model": self.base_model,
                "epochs": epochs,
                "batch_size": batch_size,
            },
            training_metrics={"num_examples": len(train_examples)},
            trained_at=datetime.utcnow(),
            is_active=False,
        )

        from ..database import db_manager
        await db_manager.save_model_metadata(metadata)

        return {
            "model_path": output_path,
            "num_examples": len(train_examples),
            "epochs": epochs,
        }

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

