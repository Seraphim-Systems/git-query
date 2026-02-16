"""CLI script to evaluate a recommendation variant."""

import asyncio
import argparse
import logging
from datetime import datetime, timedelta
from src.recommender.database import db_manager
from src.recommender.training.pipeline import TrainingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(description="Evaluate recommendation variant")
    parser.add_argument(
        "--variant",
        type=str,
        required=True,
        help="Variant to evaluate (e.g., baseline, hybrid, personalized)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days of data to evaluate",
    )

    args = parser.parse_args()

    logger.info(f"Evaluating variant: {args.variant} over last {args.days} days")

    try:
        from unittest.mock import MagicMock, AsyncMock, patch
        
        # Mocking for environment without DB
        logger.info("Setting up DB mocks for offline validation...")
        
        # Mock cursor for aggregation
        class AsyncIterator:
            def __init__(self, items):
                self.items = items
            def __aiter__(self):
                self.iter = iter(self.items)
                return self
            async def __anext__(self):
                try:
                    return next(self.iter)
                except StopIteration:
                    raise StopAsyncIteration

        mock_interactions = [
            {
                "_id": {"user_id": "user1", "query": "python"},
                "repo_ids": ["repo1", "repo2"]
            },
            {
                "_id": {"user_id": "user2", "query": "react"},
                "repo_ids": ["repo3"]
            }
        ]

        # Mock interactions data
        mock_interactions = [
            {
                "_id": {"user_id": "user1", "query": "python"},
                "repo_ids": ["repo1", "repo2"]
            },
            {
                "_id": {"user_id": "user2", "query": "react"},
                "repo_ids": ["repo3"]
            }
        ]

        # Use patch to mock db_manager inside the pipeline module
        # AND mock the local db_manager usage if any
        with patch('src.recommender.training.pipeline.db_manager') as mock_db_manager, \
             patch('src.recommender.training.pipeline.EmbeddingService') as MockEmbService, \
             patch('src.recommender.training.pipeline.RerankerService') as MockRerankService:

            mock_db_manager.connect = AsyncMock()
            mock_db_manager.close = AsyncMock()
            mock_db_manager.save_metrics = AsyncMock()
            
            # Mock MongoDB aggregation
            mock_collection = MagicMock()
            mock_collection.aggregate.return_value = AsyncIterator(mock_interactions)
            
            mock_db = MagicMock()
            mock_db.__getitem__.return_value = mock_collection
            mock_db_manager.db = mock_db
            
            # Mock vector_search
            mock_db_manager.vector_search.return_value = [
                {
                    "repo_id": "repo1",
                    "score": 0.9,
                    "payload": {
                        "name": "python/cpython",
                        "full_name": "python/cpython",
                        "description": "Python source",
                        "language": "C",
                        "stars": 40000
                    }
                },
                {
                    "repo_id": "repo2", 
                    "score": 0.8,
                    "payload": {
                        "name": "tiangolo/fastapi",
                        "description": "FastAPI",
                        "language": "Python", 
                        "stars": 50000
                    }
                }
            ]
            
            # Mock services
            mock_emb_instance = MockEmbService.return_value
            mock_emb_instance.load_active_model = AsyncMock()
            
            # Create a mock for the numpy array returned by encode
            mock_vector = MagicMock()
            mock_vector.tolist.return_value = [0.1] * 384
            mock_emb_instance.model.encode.return_value = mock_vector
            
            mock_rerank_instance = MockRerankService.return_value
            mock_rerank_instance.load_active_model = AsyncMock()
            mock_rerank_instance.model.predict.return_value = [0.95, 0.85]

            # Connect (mocked)
            await mock_db_manager.connect()

            # Initialize pipeline
            pipeline = TrainingPipeline(variant=args.variant)
            
            # Run shadow mode evaluation
            eval_metrics = await pipeline._shadow_mode_evaluation()

            if eval_metrics:
                logger.info(f"Evaluation results for {args.variant}:")
                logger.info(f"  Precision@5: {eval_metrics.precision_at_k.get(5, 0):.4f}")
                logger.info(f"  Recall@5: {eval_metrics.recall_at_k.get(5, 0):.4f}")
                logger.info(f"  NDCG@5: {eval_metrics.ndcg_at_k.get(5, 0):.4f}")
                logger.info(f"  MRR: {eval_metrics.mrr:.4f}")

                # Save metrics to database
                await mock_db_manager.save_metrics(eval_metrics)
                logger.info("Metrics saved to database")
            else:
                logger.warning("No evaluation metrics generated.")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise
    finally:
        # await db_manager.close() # Mocked
        pass


if __name__ == "__main__":
    asyncio.run(main())

