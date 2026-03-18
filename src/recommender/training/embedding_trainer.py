"""Embedding model trainer."""

import logging
from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from datetime import datetime, timezone
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
        validation_data: Dict[str, List] = None,
        early_stopping_patience: int = 2,
        resume: bool = False,
        fp16: bool = False,
        grad_accum_steps: int = 1,
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

        # Prepare training examples (and compute per-repo hashes)
        train_examples, repo_hashes = self._prepare_examples(training_data)

        if not train_examples:
            logger.warning("No training examples available")
            return {}

        # Dataset-level hash to detect unchanged training sets
        try:
            import hashlib as _hashlib
            dataset_hash = _hashlib.sha256("\n".join(sorted([h for h in repo_hashes if h])).encode()).hexdigest()
        except Exception:
            dataset_hash = ""

        # Create DataLoader
        train_dataloader = DataLoader(
            train_examples,
            shuffle=True,
            batch_size=batch_size,
        )

        # Define loss function (MultipleNegativesRankingLoss provides in-batch negatives)
        train_loss = losses.MultipleNegativesRankingLoss(self.model)

        # Train
        logger.info(f"Starting training with {len(train_examples)} examples")

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        base_output_dir = os.path.join(settings.model_path, f"embedding_{variant}_{timestamp}")
        os.makedirs(base_output_dir, exist_ok=True)

        best_metric = -1.0
        best_dir = None
        no_improve = 0

        # Simple epoch loop to allow evaluation + early stopping
        for epoch in range(epochs):
            epoch_dir = os.path.join(base_output_dir, f"epoch_{epoch+1}")
            os.makedirs(epoch_dir, exist_ok=True)

            logger.info(f"Starting epoch {epoch+1}/{epochs}")
            self.model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=1,
                warmup_steps=100,
                output_path=epoch_dir,
                show_progress_bar=True,
            )

            # Evaluate on validation set if provided
            metric_val = None
            if validation_data:
                metric_val = self._evaluate_validation(validation_data)
                logger.info(f"Validation metric after epoch {epoch+1}: {metric_val:.4f}")

            # Use validation metric for early stopping, else use epoch number
            current_metric = metric_val if metric_val is not None else (epoch + 1)

            if current_metric is not None and current_metric > best_metric:
                best_metric = current_metric
                best_dir = epoch_dir
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= early_stopping_patience:
                logger.info("Early stopping triggered")
                break

        # Choose best_dir if available else last epoch_dir
        output_path = best_dir or epoch_dir
        logger.info(f"Model saved to: {output_path}")

        # Save metadata via registry
        metadata = ModelMetadata(
            model_id=f"embedding_{variant}_{datetime.now(timezone.utc).timestamp()}",
            model_type="embedding",
            variant=variant,
            version="1.0.0",
            path=os.path.relpath(output_path, settings.model_path),
            hyperparameters={
                "base_model": self.base_model,
                "epochs": epochs,
                "batch_size": batch_size,
            },
            metrics={
                "num_examples": float(len(train_examples)),
                "best_metric": float(best_metric)
            },
            trained_at=datetime.now(timezone.utc),
            is_active=False,
            status="candidate"
        )

        from ..services.registry_service import ModelRegistryService
        registry = ModelRegistryService()
        await registry.register_model(metadata)

        # Persist training metadata locally for incremental checks
        try:
            meta_dir = os.path.join(settings.model_path, "metadata")
            os.makedirs(meta_dir, exist_ok=True)
            training_meta = {
                "timestamp": timestamp,
                "num_examples": len(train_examples),
                "dataset_hash": dataset_hash,
                "best_metric": best_metric,
                "variant": variant,
            }
            meta_path = os.path.join(meta_dir, f"training_metadata_{timestamp}.json")
            with open(meta_path, 'w') as f:
                import json
                json.dump(training_meta, f, indent=2, default=str)

            latest_meta = os.path.join(meta_dir, "training_metadata_latest.json")
            tmp = latest_meta + ".tmp"
            with open(tmp, 'w') as f:
                json.dump(training_meta, f, indent=2, default=str)
            os.replace(tmp, latest_meta)
        except Exception:
            logger.exception("Could not write local training metadata")

        return {
            "model_path": output_path,
            "num_examples": len(train_examples),
            "epochs": epochs,
        }

    def _prepare_examples(self, training_data: Dict[str, List]) -> (List[InputExample], List[str]):
        """
        Prepare training examples for sentence-transformers.

        Creates (query, positive_doc) pairs for training.
        """
        examples = []
        hashes = []

        queries = training_data.get("queries", [])
        positive_repos = training_data.get("positive_repos", [])

        for query, positive_repo in zip(queries, positive_repos):
            # Create text representation of repo
            repo_text = self._repo_to_text(positive_repo)

            # Create training example
            example = InputExample(texts=[query, repo_text])
            examples.append(example)

            # Compute per-repo text hash for incremental checks
            try:
                import hashlib as _hashlib
                h = _hashlib.sha256(repo_text.encode('utf-8')).hexdigest()
            except Exception:
                h = ""
            hashes.append(h)

        return examples, hashes

    def _repo_to_text(self, repo: Dict[str, Any]) -> str:
        """Convert repository data to text for embedding."""
        from .utils import prepare_repo_text
        return prepare_repo_text(repo)

    def _evaluate_validation(self, validation_data: Dict[str, List]) -> float:
        """Simple evaluation computing MRR@10 on the validation set."""
        try:
            queries = validation_data.get("queries", [])
            positive_repos = validation_data.get("positive_repos", [])
            if not queries or not positive_repos:
                return 0.0

            # Embed candidates (unique)
            cand_texts = [self._repo_to_text(r) for r in positive_repos]
            import numpy as _np
            cand_emb = self.model.encode(cand_texts, convert_to_numpy=True, normalize_embeddings=True)

            # Embed queries
            q_emb = self.model.encode(queries, convert_to_numpy=True, normalize_embeddings=True)

            # Compute MRR@10
            rr_sum = 0.0
            for qi, qv in enumerate(q_emb):
                sims = _np.dot(cand_emb, qv)
                # get ranking (higher is better)
                ranks = _np.argsort(-sims)
                # find index of the positive repo (assumes same order)
                # If multiple positives exist this is simplistic but works for our checks
                pos_idx = qi if qi < len(positive_repos) else 0
                rank = int(_np.where(ranks == pos_idx)[0][0]) + 1
                if rank <= 10:
                    rr_sum += 1.0 / rank

            mrr = rr_sum / len(q_emb)
            return float(mrr)
        except Exception:
            logger.exception("Validation evaluation failed")
            return 0.0

