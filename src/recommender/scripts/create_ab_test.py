"""CLI script to create an A/B test configuration in the database."""

import asyncio
import argparse
import logging
from datetime import datetime, timedelta
from ..database import db_manager
from ..models import ABTestConfig
from ..config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(description="Create A/B test configuration")
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Name of the A/B test",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Description of the A/B test",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        required=True,
        help="List of variant names (e.g., baseline hybrid)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        type=float,
        required=True,
        help="Traffic splits (e.g., 0.5 0.5 for 50/50 split)",
    )
    parser.add_argument(
        "--duration-days",
        type=int,
        default=14,
        help="Test duration in days",
    )

    args = parser.parse_args()

    if len(args.variants) != len(args.splits):
        raise ValueError("Number of variants must match number of splits")

    if abs(sum(args.splits) - 1.0) > 0.001:
        raise ValueError("Traffic splits must sum to 1.0")

    test_id = f"ab_test_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    ab_test = ABTestConfig(
        test_id=test_id,
        name=args.name,
        description=args.description,
        variants=args.variants,
        traffic_split=dict(zip(args.variants, args.splits)),
        start_date=datetime.utcnow(),
        end_date=datetime.utcnow() + timedelta(days=args.duration_days),
        is_active=True,
    )

    logger.info(f"Creating A/B test: {args.name}")
    logger.info(f"  Variants: {args.variants}")
    logger.info(f"  Traffic split: {dict(zip(args.variants, args.splits))}")
    logger.info(f"  Duration: {args.duration_days} days")

    try:
        await db_manager.connect()

        # Deactivate any existing active tests
        await db_manager.db[settings.ab_tests_collection].update_many(
            {"is_active": True},
            {"$set": {"is_active": False}},
        )

        # Insert new test
        await db_manager.db[settings.ab_tests_collection].insert_one(
            ab_test.model_dump()
        )

        logger.info(f"A/B test created successfully: {test_id}")

    except Exception as e:
        logger.error(f"Failed to create A/B test: {e}")
        raise
    finally:
        await db_manager.close()


if __name__ == "__main__":
    asyncio.run(main())
