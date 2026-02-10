"""CLI script to run training pipeline."""

import asyncio
import argparse
import logging
from recommender.training import TrainingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(description="Run recommendation model training pipeline")
    parser.add_argument(
        "--variant",
        type=str,
        default="default",
        help="Model variant name (e.g., v1, v2, experimental)",
    )
    parser.add_argument(
        "--train-embeddings",
        action="store_true",
        default=True,
        help="Train embedding model",
    )
    parser.add_argument(
        "--train-reranker",
        action="store_true",
        default=True,
        help="Train reranker model",
    )
    parser.add_argument(
        "--min-interactions",
        type=int,
        default=1000,
        help="Minimum interactions required for training",
    )

    args = parser.parse_args()

    logger.info(f"Starting training pipeline for variant: {args.variant}")

    pipeline = TrainingPipeline(variant=args.variant)

    try:
        await pipeline.run_full_pipeline(
            train_embeddings=args.train_embeddings,
            train_reranker=args.train_reranker,
            min_interactions=args.min_interactions,
        )
        logger.info("Training pipeline completed successfully!")
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())

