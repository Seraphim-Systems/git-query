"""A/B testing service for comparing recommendation variants."""

import hashlib
import random
from typing import Optional
from ..models import ABTestConfig
from ..database import db_manager
from ..config import settings


class ABTestService:
    """
    Service for managing A/B tests.

    Uses consistent hashing to assign users to variants.
    """

    async def get_variant_for_user(
        self, user_id: Optional[str], request_variant: Optional[str] = None
    ) -> str:
        """
        Get the variant for a user.

        Args:
            user_id: User identifier (can be None for anonymous)
            request_variant: Explicitly requested variant (overrides A/B test)

        Returns:
            Variant name to use
        """
        # If variant explicitly requested, use it
        if request_variant:
            return request_variant

        # If A/B testing disabled, use default
        if not settings.ab_test_enabled:
            return settings.default_variant

        # Get active A/B test
        ab_test = await db_manager.get_active_ab_test()

        if not ab_test:
            return settings.default_variant

        # Assign variant using consistent hashing
        if user_id:
            variant = self._hash_user_to_variant(user_id, ab_test)
        else:
            # Random assignment for anonymous users
            variant = random.choices(
                list(ab_test.variants),
                weights=[ab_test.traffic_split.get(v, 0) for v in ab_test.variants],
            )[0]

        return variant

    def _hash_user_to_variant(self, user_id: str, ab_test: ABTestConfig) -> str:
        """
        Use consistent hashing to assign user to variant.

        This ensures the same user always gets the same variant.
        """
        # Create hash of user_id + test_id
        hash_input = f"{user_id}:{ab_test.test_id}"
        hash_value = int(hashlib.sha256(hash_input.encode()).hexdigest(), 16)

        # Normalize to [0, 1]
        normalized = (hash_value % 10000) / 10000.0

        # Assign to variant based on traffic split
        cumulative = 0.0
        for variant in ab_test.variants:
            cumulative += ab_test.traffic_split.get(variant, 0)
            if normalized <= cumulative:
                return variant

        # Fallback (should not happen if traffic splits sum to 1)
        return ab_test.variants[0]

