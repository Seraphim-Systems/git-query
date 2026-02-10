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

    # Create A/B test config
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
        await db_manager.db[db_manager.db.ab_tests_collection].update_many(
            {"is_active": True},
            {"$set": {"is_active": False}},
        )

        # Insert new test
        await db_manager.db["ab_tests"].insert_one(ab_test.model_dump())

        logger.info(f"A/B test created successfully: {test_id}")

    except Exception as e:
        logger.error(f"Failed to create A/B test: {e}")
        raise
    finally:
        await db_manager.close()


if __name__ == "__main__":
    asyncio.run(main())
"""Git-Query Recommendation System.

AI-powered repository recommendation with hybrid retrieval,
personalization, and A/B testing.
"""

__version__ = "1.0.0"

from .config import settings
from .models import (
    RecommendationRequest,
    RecommendationResponse,
    UserInteraction,
    InteractionType,
)
from .engines import (
    RecommendationEngine,
    BaselineEngine,
    HybridRetrievalEngine,
    PersonalizedEngine,
)

__all__ = [
    "settings",
    "RecommendationRequest",
    "RecommendationResponse",
    "UserInteraction",
    "InteractionType",
    "RecommendationEngine",
    "BaselineEngine",
    "HybridRetrievalEngine",
    "PersonalizedEngine",
]

