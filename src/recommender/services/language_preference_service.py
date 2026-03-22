"""Service for managing explicit and implicit language preferences."""

import math
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from ..config import settings
from ..database import db_manager
from ..models import UserPreferences

logger = logging.getLogger(__name__)


class LanguagePreferenceService:
    """
    Manages user language preferences from two sources:

    Implicit: inferred from interactions with repos in specific languages
                (handled by PersonalizationService + time decay here)
    Explicit: directly stated by the user via the preferences API
    """

    # Public API
    async def set_explicit_languages(
        self, user_id: str, languages: List[str]
    ) -> UserPreferences:
        """
        Store languages the user has explicitly said they work in.

        Each declared language gets a score floor of `explicit_language_boost`
        so it always appears near the top of their ranked preferences even if
        they haven't interacted with repos in that language yet.
        """
        prefs = await self._get_or_create_prefs(user_id)

        normalized = [lang.strip().lower() for lang in languages if lang.strip()]
        prefs.explicit_languages = normalized

        for lang in normalized:
            current = prefs.language_preferences.get(lang, 0.0)
            prefs.language_preferences[lang] = max(
                current, settings.explicit_language_boost
            )

        prefs.last_updated = datetime.now(timezone.utc)
        await db_manager.update_user_preferences(prefs)
        logger.info("Set explicit languages for user %s: %s", user_id, normalized)
        return prefs

    async def get_top_languages(
        self, user_id: str, top_n: int = 5
    ) -> List[Dict]:
        """
        Return the user's top N preferred languages with source info.

        Each entry looks like:
            {"language": "python", "score": 0.92, "explicit": True}
        """
        prefs = await db_manager.get_user_preferences(user_id)
        if not prefs:
            return []

        explicit_set = set(prefs.explicit_languages or [])
        ranked = sorted(
            prefs.language_preferences.items(),
            key=lambda kv: kv[1],
            reverse=True,
        )[:top_n]

        return [
            {
                "language": lang,
                "score": round(score, 4),
                "explicit": lang in explicit_set,
            }
            for lang, score in ranked
        ]

    async def remove_language(
        self, user_id: str, language: str
    ) -> UserPreferences:
        """
        Remove a language from both explicit declarations and scored preferences.
        Raises ValueError if the user has no preferences on record.
        """
        prefs = await db_manager.get_user_preferences(user_id)
        if not prefs:
            raise ValueError(f"No preferences found for user '{user_id}'")

        lang = language.strip().lower()

        if lang in (prefs.explicit_languages or []):
            prefs.explicit_languages.remove(lang)

        prefs.language_preferences.pop(lang, None)
        prefs.last_updated = datetime.now(timezone.utc)

        await db_manager.update_user_preferences(prefs)
        logger.info("Removed language '%s' from user %s", lang, user_id)
        return prefs

    # Helpers used by PersonalizationService
    @staticmethod
    def apply_time_decay(signal_weight: float, interaction_time: datetime) -> float:
        """
        Exponential time decay: weight halves every `language_decay_half_life_days`.

        Formula:  w * 2^(−age / half_life)
        """
        now = datetime.now(timezone.utc)
        if interaction_time.tzinfo is None:
            interaction_time = interaction_time.replace(tzinfo=timezone.utc)

        age_days = max(0.0, (now - interaction_time).total_seconds() / 86_400)
        half_life = settings.language_decay_half_life_days
        decay = math.pow(2, -age_days / half_life)
        return signal_weight * decay

    # Internal
    @staticmethod
    async def _get_or_create_prefs(user_id: str) -> UserPreferences:
        prefs = await db_manager.get_user_preferences(user_id)
        if prefs is None:
            prefs = UserPreferences(
                user_id=user_id,
                language_preferences={},
                topic_preferences={},
                total_interactions=0,
            )
        return prefs