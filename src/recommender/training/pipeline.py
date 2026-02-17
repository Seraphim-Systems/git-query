"""Training pipeline orchestrator."""

import asyncio
import os
import json
from datetime import datetime
from typing import Optional, Dict, List, Any
import logging

from ..config import settings
from ..database import db_manager
from ..models import ModelMetadata, EvaluationMetrics, RepositoryResult
from ..services import ModelRegistryService, EmbeddingService, RerankerService
from .embedding_trainer import EmbeddingTrainer
from .reranker_trainer import RerankerTrainer
from .evaluator import RecommenderEvaluator

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
        
        # Ensure model directories exist
        os.makedirs(settings.model_path, exist_ok=True)
        os.makedirs(settings.checkpoint_path, exist_ok=True)
        os.makedirs(settings.eval_path, exist_ok=True)
        os.makedirs(os.path.join(settings.model_path, "registry"), exist_ok=True)

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

    async def _prepare_validation_data(self, limit: int = 1000) -> Dict[tuple, List[str]]:
        """
        Extract interactions where action='click' from MongoDB.

        Args:
            limit: Maximum number of interactions to process

        Returns:
            Dictionary mapping (user_id, query) to list of relevant repo_ids
        """
        logger.info(f"Extracting validation data (limit={limit})...")
        
        # Use aggregation to group by user_id and query efficiently
        pipeline = [
            {"$match": {"interaction_type": "click"}},
            {"$sort": {"timestamp": -1}},
            {"$limit": limit},
            {
                "$group": {
                    "_id": {"user_id": "$user_id", "query": "$query"},
                    "repo_ids": {"$addToSet": "$repo_id"}
                }
            }
        ]

        cursor = db_manager.db[settings.interactions_collection].aggregate(pipeline)
        
        validation_data = {}
        async for doc in cursor:
            key = (doc["_id"]["user_id"], doc["_id"]["query"])
            validation_data[key] = doc["repo_ids"]
            
        logger.info(f"Extracted {len(validation_data)} unique query sessions for validation")
        return validation_data

    async def _shadow_mode_evaluation(self) -> EvaluationMetrics:
        """
        Evaluate new models in shadow mode.

        Shadow mode: Generate recommendations with new model
        but don't show to users. Compare with actual user choices.
        """
        logger.info("Starting shadow mode evaluation...")
        
        # 1. Prepare validation data
        validation_data = await self._prepare_validation_data()
        
        predictions = []
        ground_truth = []
        start_time = datetime.utcnow()

        if validation_data:
            # 2. Initialize models/services
            # Use trained models if available, otherwise load active models
            embedding_model = self.embedding_trainer.model
            if not embedding_model:
                logger.info("Loading active embedding model for evaluation...")
                emb_service = EmbeddingService()
                await emb_service.load_active_model(self.variant)
                embedding_model = emb_service.model
                
            reranker_model = self.reranker_trainer.model
            if not reranker_model:
                logger.info("Loading active reranker model for evaluation...")
                reranker_service = RerankerService()
                await reranker_service.load_active_model(self.variant)
                reranker_model = reranker_service.model

            # 3. Run evaluation loop
            # Limit evaluation to avoid long runtime during shadow mode
            MAX_EVAL_SAMPLES = 100
            processed_count = 0
            
            for (user_id, query), relevant_repos in validation_data.items():
                if processed_count >= MAX_EVAL_SAMPLES:
                    break
                    
                try:
                    # Embed query
                    if hasattr(embedding_model, "encode"):
                        query_vector = embedding_model.encode(query).tolist()
                    else:
                        logger.warning("Embedding model does not have 'encode' method")
                        continue
                    
                    # Retrieve candidates (Top K * 2 for reranking)
                    hits = db_manager.vector_search(
                        query_vector=query_vector,
                        top_k=settings.rerank_top_k * 2
                    )
                    
                    if not hits:
                        predictions.append([])
                        ground_truth.append(relevant_repos)
                        processed_count += 1
                        continue
                    
                    # Convert to candidate objects for reranking
                    candidates = []
                    for hit in hits:
                        payload = hit.get("payload", {})
                        candidates.append(RepositoryResult(
                            repo_id=hit["repo_id"],
                            score=hit["score"],
                            rank=0,
                            name=payload.get("name", "Unknown"),
                            full_name=payload.get("full_name", "Unknown"),
                            description=payload.get("description", ""),
                            language=payload.get("language"),
                            stars=payload.get("stars", 0),
                            forks=payload.get("forks", 0),
                            url=payload.get("url", ""),
                            license=payload.get("license"),
                            last_updated=None
                        ))
                    
                    # Rerank
                    pairs = []
                    for cand in candidates:
                        parts = [cand.name]
                        if cand.description:
                            parts.append(cand.description)
                        if cand.language:
                            parts.append(f"Language: {cand.language}")
                        text = " | ".join(parts)
                        pairs.append([query, text])
                    
                    if pairs and hasattr(reranker_model, "predict"):
                        scores = reranker_model.predict(pairs)
                        
                        # Sort candidates by new score
                        scored_candidates = sorted(
                            zip(candidates, scores),
                            key=lambda x: x[1],
                            reverse=True
                        )
                        
                        # Take top K
                        top_k_ids = [c.repo_id for c, s in scored_candidates[:settings.final_top_k]]
                    else:
                        top_k_ids = [c.repo_id for c in candidates[:settings.final_top_k]]
                        
                    predictions.append(top_k_ids)
                    ground_truth.append(relevant_repos)
                    processed_count += 1
                    
                except Exception as e:
                    logger.error(f"Error evaluating query '{query}': {e}")
                    continue
        else:
            logger.warning("No validation data found. Returning empty metrics.")

        # 4. Compute metrics
        evaluator = RecommenderEvaluator()
        metrics = evaluator.evaluate(predictions, ground_truth)
        
        # 5. Save report
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        report_filename = f"{self.variant}_{timestamp}.json"
        report_path = os.path.join(settings.eval_path, report_filename)
        
        evaluator.save_report(metrics, report_path)
        logger.info(f"Evaluation report saved to {report_path}")

        # 6. Return EvaluationMetrics object
        return EvaluationMetrics(
            variant=self.variant,
            precision_at_k=metrics.get("precision_at_k", {}),
            recall_at_k=metrics.get("recall_at_k", {}),
            ndcg_at_k=metrics.get("ndcg_at_k", {}),
            mrr=metrics.get("mrr", 0.0),
            click_through_rate=0.0,
            avg_response_time_ms=0.0,
            total_queries=len(predictions),
            total_interactions=sum(len(gt) for gt in ground_truth),
            evaluation_period_start=start_time,
            evaluation_period_end=datetime.utcnow(),
        )

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

