"""CLI script to evaluate a recommendation variant."""

import asyncio
import argparse
import logging
from datetime import datetime, timedelta
from recommender.database import db_manager
from recommender.training.evaluator import RecommenderEvaluator

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
        await db_manager.connect()

        # Get interactions for the variant
        cutoff = datetime.utcnow() - timedelta(days=args.days)

        # This is a placeholder - you'd need to implement the actual query
        # to get interactions and compute metrics

        evaluator = RecommenderEvaluator()

        # Example evaluation (would need real data)
        predictions = []  # List of [repo_id, ...] for each query
        ground_truth = []  # List of [clicked_repo_id, ...] for each query

        metrics = evaluator.evaluate(predictions, ground_truth)

        logger.info(f"Evaluation results for {args.variant}:")
        logger.info(f"  Precision@5: {metrics['precision_at_k'].get(5, 0):.4f}")
        logger.info(f"  Recall@5: {metrics['recall_at_k'].get(5, 0):.4f}")
        logger.info(f"  NDCG@5: {metrics['ndcg_at_k'].get(5, 0):.4f}")
        logger.info(f"  MRR: {metrics['mrr']:.4f}")

        # Save metrics to database
        from recommender.models import EvaluationMetrics

        eval_metrics = EvaluationMetrics(
            variant=args.variant,
            precision_at_k=metrics["precision_at_k"],
            recall_at_k=metrics["recall_at_k"],
            ndcg_at_k=metrics["ndcg_at_k"],
            mrr=metrics["mrr"],
            click_through_rate=0.0,  # Would compute from data
            avg_response_time_ms=0.0,  # Would compute from data
            total_queries=len(predictions),
            total_interactions=0,
            evaluation_period_start=cutoff,
            evaluation_period_end=datetime.utcnow(),
        )

        await db_manager.save_metrics(eval_metrics)
        logger.info("Metrics saved to database")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise
    finally:
        await db_manager.close()


if __name__ == "__main__":
    asyncio.run(main())

