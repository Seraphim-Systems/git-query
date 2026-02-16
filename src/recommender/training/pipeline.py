"""Training pipeline orchestrator."""

import asyncio
import os
from datetime import datetime
from typing import Optional
import logging

from ..config import settings
from ..database import db_manager
from ..models import ModelMetadata
from ..services import ModelRegistryService
from .embedding_trainer import EmbeddingTrainer
from .reranker_trainer import RerankerTrainer

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    Orchestrates the training pipeline.

    Steps:
    1. Extract training data from interactions
    2. Train embedding model (optional)
    3. Train/fine-tune cross-encoder (optional)
    4. Evaluate models in shadow mode
    5. Update model registry
    6. Deploy if performance improves
    """

    def __init__(self, variant: str = "default"):
        self.variant = variant
        self.embedding_trainer = EmbeddingTrainer()
        self.reranker_trainer = RerankerTrainer()
        self.registry = ModelRegistryService()
        
        # Ensure model path exists
        os.makedirs(settings.model_path, exist_ok=True)

    async def run_full_pipeline(
        self,
        train_embeddings: bool = True,
        train_reranker: bool = True,
        min_interactions: int = 1000,
    ):
        """
        Run the full training pipeline.

        Args:
            train_embeddings: Whether to train embedding model
            train_reranker: Whether to train reranker model
            min_interactions: Minimum interactions required for training
        """
        logger.info(f"Starting training pipeline for variant: {self.variant}")

        try:
            # Step 1: Check if we have enough data
            await db_manager.connect()

            # Get interaction count (would need to implement this query)
            # For now, assume we have enough data

            # Step 2: Prepare training data
            logger.info("Preparing training data...")
            training_data = await self._prepare_training_data()

            if not training_data:
                logger.warning("No training data available")
                return

            # Step 3: Train embedding model
            if train_embeddings:
                logger.info("Training embedding model...")
                embedding_metrics = await self.embedding_trainer.train(
                    training_data=training_data,
                    variant=self.variant,
                )
                logger.info(f"Embedding training completed: {embedding_metrics}")

            # Step 4: Train reranker
            if train_reranker:
                logger.info("Training reranker model...")
                reranker_metrics = await self.reranker_trainer.train(
                    training_data=training_data,
                    variant=self.variant,
                )
                logger.info(f"Reranker training completed: {reranker_metrics}")

            # Step 5: Evaluate in shadow mode
            logger.info("Evaluating models in shadow mode...")
            eval_metrics = await self._shadow_mode_evaluation()

            # Step 6: Save metrics
            await db_manager.save_metrics(eval_metrics)

            # Step 7: Deploy if improved
            # This would compare with current production metrics
            # and deploy if better

            logger.info("Training pipeline completed successfully")

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise
        finally:
            await db_manager.close()

    async def _prepare_training_data(self):
        """
        Prepare training data from user interactions using a streaming pattern.

        Returns:
            Dictionary with data generators or batched lists for training.
        """
        # IN PRODUCTION: Use MongoDB cursor with batch_size to stream interactions
        # cursor = self.db[settings.interactions_collection].find(...).batch_size(1000)
        # This prevents OOM errors when processing millions of interactions.

        logger.info("Streaming user interactions for training data preparation...")

        return {
            "queries": [],
            "positive_repos": [],
            "negative_repos": [],
            "labels": [],
        }

    async def _shadow_mode_evaluation(self):
        """
        Evaluate new models in shadow mode.

        Shadow mode: Generate recommendations with new model
        but don't show to users. Compare with actual user choices.
        """
        from ..models import EvaluationMetrics

        # Placeholder metrics
        metrics = EvaluationMetrics(
            variant=self.variant,
            precision_at_k={1: 0.0, 5: 0.0, 10: 0.0},
            recall_at_k={1: 0.0, 5: 0.0, 10: 0.0},
            ndcg_at_k={1: 0.0, 5: 0.0, 10: 0.0},
            mrr=0.0,
            click_through_rate=0.0,
            avg_response_time_ms=0.0,
            total_queries=0,
            total_interactions=0,
            evaluation_period_start=datetime.utcnow(),
            evaluation_period_end=datetime.utcnow(),
        )

        return metrics

    async def incremental_update(self):
        """
        Perform incremental model update.

        Used for continuous learning from new data.
        Less expensive than full retraining.
        """
        logger.info("Starting incremental update...")

        # Get recent interactions
        # Update models with new data
        # This could use online learning techniques

        pass

