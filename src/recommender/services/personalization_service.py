"""Service for learning and updating user preferences."""

from typing import Dict
from collections import defaultdict
from ..models import UserInteraction, UserPreferences, InteractionType
from ..database import db_manager
from datetime import datetime, timezone


class PersonalizationService:
    """
    Service for learning user preferences from interactions.

    Uses implicit feedback signals:
    - Clicks, saves, thumbs_up = positive signal
    - Dismissals, thumbs_down = negative signal
    """

    POSITIVE_SIGNALS = {
        InteractionType.CLICK: 1.0,
        InteractionType.SAVE: 2.0,
        InteractionType.THUMBS_UP: 3.0,
    }

    NEGATIVE_SIGNALS = {
        InteractionType.DISMISS: -1.0,
        InteractionType.THUMBS_DOWN: -2.0,
    }

    async def update_preferences_from_interaction(
        self, interaction: UserInteraction, repo_data: Dict
    ):
        """
        Update user preferences based on a new interaction.

        Args:
            interaction: The user interaction event
            repo_data: Data about the repository interacted with
        """
        # Get current preferences
        prefs = await db_manager.get_user_preferences(interaction.user_id)

        if prefs is None:
            prefs = UserPreferences(
                user_id=interaction.user_id,
                language_preferences={},
                topic_preferences={},
                total_interactions=0,
            )

        # Get signal weight
        signal_weight = self._get_signal_weight(interaction.interaction_type)

        # Update language preferences
        if repo_data.get("language"):
            language = repo_data["language"]
            current = prefs.language_preferences.get(language, 0.0)
            prefs.language_preferences[language] = current + signal_weight

        # Update topic preferences
        if repo_data.get("topics"):
            for topic in repo_data["topics"]:
                current = prefs.topic_preferences.get(topic, 0.0)
                prefs.topic_preferences[topic] = current + signal_weight

        # Normalize preferences to prevent unbounded growth
        prefs.language_preferences = self._normalize_preferences(
            prefs.language_preferences
        )
        prefs.topic_preferences = self._normalize_preferences(
            prefs.topic_preferences
        )

        # Update metadata
        prefs.total_interactions += 1
        prefs.last_updated = datetime.now(timezone.utc)

        # Save
        await db_manager.update_user_preferences(prefs)

    async def batch_update_preferences(self, user_id: str):
        """
        Batch update user preferences from all recent interactions.

        This can be run periodically to recompute preferences.
        """
        interactions = await db_manager.get_user_interactions(user_id, limit=1000)

        language_scores = defaultdict(float)
        topic_scores = defaultdict(float)

        for interaction in interactions:
            signal_weight = self._get_signal_weight(interaction.interaction_type)

            # Use metadata stored with the interaction if available
            if hasattr(interaction, 'metadata') and interaction.metadata:
                meta = interaction.metadata
                if meta.get("language"):
                    language_scores[meta["language"]] += signal_weight
                for topic in meta.get("topics", []):
                    topic_scores[topic] += signal_weight

        # Create or update preferences
        prefs = UserPreferences(
            user_id=user_id,
            language_preferences=self._normalize_preferences(dict(language_scores)),
            topic_preferences=self._normalize_preferences(dict(topic_scores)),
            total_interactions=len(interactions),
            last_updated=datetime.now(timezone.utc),
        )

        await db_manager.update_user_preferences(prefs)

    def _get_signal_weight(self, interaction_type: InteractionType) -> float:
        """Get the weight for an interaction type."""
        if interaction_type in self.POSITIVE_SIGNALS:
            return self.POSITIVE_SIGNALS[interaction_type]
        elif interaction_type in self.NEGATIVE_SIGNALS:
            return self.NEGATIVE_SIGNALS[interaction_type]
        else:
            return 0.0

    def _normalize_preferences(self, preferences: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize preferences to [0, 1] range.

        Uses min-max normalization with smoothing.
        """
        if not preferences:
            return {}

        values = list(preferences.values())
        min_val = min(values)
        max_val = max(values)

        if max_val == min_val:
            return {k: 0.5 for k in preferences}

        normalized = {}
        for key, value in preferences.items():
            # Shift to positive range
            shifted = value - min_val
            # Normalize to [0, 1]
            normalized[key] = shifted / (max_val - min_val)

        return normalized

