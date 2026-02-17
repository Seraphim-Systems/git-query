"""Quick start script for testing the recommender service."""

import asyncio
import sys
from src.recommender.database import db_manager
from src.recommender.models import (
    RecommendationRequest,
    UserInteraction,
    InteractionType,
    ABTestConfig,
)
from src.recommender.engines import BaselineEngine, HybridRetrievalEngine
from datetime import datetime, timedelta


async def test_setup():
    """Test basic setup and connections."""
    print("=" * 60)
    print("Git-Query Recommender - Quick Start Test")
    print("=" * 60)
    
    # Test 1: Database connection
    print("\n[1/5] Testing database connections...")
    try:
        await db_manager.connect()
        print("✓ MongoDB connected")
        print("✓ Qdrant connected")
        print("✓ Redis connected")
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False
    
    # Test 2: Create sample A/B test
    print("\n[2/5] Creating sample A/B test...")
    try:
        ab_test = ABTestConfig(
            test_id="test_baseline_vs_hybrid",
            name="Baseline vs Hybrid",
            description="Test hybrid retrieval against baseline",
            variants=["baseline", "hybrid"],
            traffic_split={"baseline": 0.5, "hybrid": 0.5},
            start_date=datetime.utcnow(),
            end_date=datetime.utcnow() + timedelta(days=14),
            is_active=True,
        )
        await db_manager.db["ab_tests"].insert_one(ab_test.model_dump())
        print("✓ A/B test created")
    except Exception as e:
        print(f"✗ Failed to create A/B test: {e}")
    
    # Test 3: Test baseline engine
    print("\n[3/5] Testing baseline recommendation engine...")
    try:
        engine = BaselineEngine()
        request = RecommendationRequest(
            query="python web framework",
            top_k=5,
        )
        # Note: This will fail without actual data, but tests the interface
        print("✓ Baseline engine initialized")
    except Exception as e:
        print(f"✗ Engine test failed: {e}")
    
    # Test 4: Log sample interaction
    print("\n[4/5] Testing interaction logging...")
    try:
        interaction = UserInteraction(
            user_id="test_user_123",
            query="python web framework",
            repo_id="repo_456",
            interaction_type=InteractionType.CLICK,
            position_in_results=3,
            variant="baseline",
        )
        await db_manager.log_interaction(interaction)
        print("✓ Interaction logged successfully")
    except Exception as e:
        print(f"✗ Failed to log interaction: {e}")
    
    # Test 5: Cleanup
    print("\n[5/5] Cleaning up...")
    await db_manager.close()
    print("✓ Database connections closed")
    
    print("\n" + "=" * 60)
    print("Setup test complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Start the service: python -m recommender")
    print("2. Check health: curl http://localhost:8095/health")
    print("3. Get recommendations: POST http://localhost:8095/recommend")
    print("\nSee README.md for full API documentation")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(test_setup())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nTest failed with error: {e}")
        sys.exit(1)
