"""Embedding model trainer."""

import logging
from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from datetime import datetime
import os

from ..config import settings
from ..models import ModelMetadata

logger = logging.getLogger(__name__)


class EmbeddingTrainer:
    """
    Trainer for semantic embedding models.

    Uses sentence-transformers library for training bi-encoders.
    Can fine-tune existing models on domain-specific data.
    """

    def __init__(self, base_model: str = None):
        self.base_model = base_model or settings.embedding_model_name
        self.model = None

    async def train(
        self,
        training_data: Dict[str, List],
        variant: str,
        epochs: int = 3,
        batch_size: int = 16,
    ) -> Dict[str, Any]:
        """
        Train/fine-tune embedding model.

        Args:
            training_data: Dictionary with queries and positive/negative examples
            variant: Model variant name
            epochs: Number of training epochs
            batch_size: Training batch size

        Returns:
            Training metrics
        """
        logger.info(f"Training embedding model: {self.base_model}")

        # Load base model
        self.model = SentenceTransformer(self.base_model)

        # Prepare training examples
        train_examples = self._prepare_examples(training_data)

        if not train_examples:
            logger.warning("No training examples available")
            return {}

        # Create DataLoader
        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=batch_size,
        )

        # Define loss function
        # Using MultipleNegativesRankingLoss for semantic search
        train_loss = losses.MultipleNegativesRankingLoss(self.model)

        # Train
        logger.info(f"Starting training with {len(train_examples)} examples")

        output_path = os.path.join(
            settings.model_path,
            f"embedding_{variant}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        )

        self.model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=100,
            output_path=output_path,
            show_progress_bar=True,
        )

        logger.info(f"Model saved to: {output_path}")

        # Save metadata
        metadata = ModelMetadata(
            model_id=f"embedding_{variant}_{datetime.utcnow().timestamp()}",
            model_type="embedding",
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
            is_active=False,  # Set to True after validation
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
        Prepare training examples for sentence-transformers.

        Creates (query, positive_doc) pairs for training.
        """
        examples = []

        queries = training_data.get("queries", [])
        positive_repos = training_data.get("positive_repos", [])

        for query, positive_repo in zip(queries, positive_repos):
            # Create text representation of repo
            repo_text = self._repo_to_text(positive_repo)

            # Create training example
            example = InputExample(texts=[query, repo_text])
            examples.append(example)

        return examples

    def _repo_to_text(self, repo: Dict[str, Any]) -> str:
        """Convert repository data to text for embedding."""
        parts = []

        if repo.get("name"):
            parts.append(repo["name"])

        if repo.get("description"):
            parts.append(repo["description"])

        if repo.get("topics"):
            parts.append(" ".join(repo["topics"]))

        return " | ".join(parts)

