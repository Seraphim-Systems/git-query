"""Data transformation pipeline - normalizes raw repository documents."""

import logging
import re
from typing import Dict, Any, Optional
from datetime import datetime

from processing.config import settings

logger = logging.getLogger(__name__)


class DataTransformer:
    """Transforms raw repository data into a consistent cleaned schema."""

    def __init__(self):
        self.max_desc_length = settings.max_description_length

    def transform(self, raw_record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Transform and clean a raw record.

        Args:
            raw_record: Raw repository data

        Returns:
            Cleaned record or None if invalid
        """
        try:
            # Keep all raw fields by default, then normalize required ones.
            cleaned = dict(raw_record)

            full_name = self._first_non_empty(
                raw_record.get("full_name"),
                raw_record.get("nameWithOwner"),
            )
            full_name = self._clean_string(full_name)

            name = self._first_non_empty(
                raw_record.get("name"),
                self._extract_repo_name(full_name),
            )
            name = self._clean_string(name)

            owner = self._first_non_empty(
                raw_record.get("owner"),
                raw_record.get("owner_login"),
                self._extract_owner(full_name),
            )
            owner = self._clean_string(owner)

            language = self._first_non_empty(
                raw_record.get("language"),
                raw_record.get("primaryLanguage"),
                "unknown",
            )
            language = self._clean_string(language) or "unknown"

            stars = self._clean_integer(
                self._first_non_empty(raw_record.get("stars"), raw_record.get("stargazers_count"), 0)
            )
            forks = self._clean_integer(
                self._first_non_empty(raw_record.get("forks"), raw_record.get("forks_count"), 0)
            )

            repo_id = self._first_non_empty(
                raw_record.get("repo_id"),
                raw_record.get("id"),
                full_name,
            )
            repo_id = self._clean_string(repo_id)

            description = self._clean_description(raw_record.get("description"))
            topics = self._clean_topics(raw_record.get("topics"))
            license_value = self._normalize_license(raw_record.get("license"))

            updated_at = self._to_iso_timestamp(
                self._first_non_empty(raw_record.get("updated_at"), raw_record.get("updatedAt"))
            )
            pushed_at = self._to_iso_timestamp(
                self._first_non_empty(raw_record.get("pushed_at"), raw_record.get("pushedAt"))
            )

            cleaned.update(
                {
                    "repo_id": repo_id,
                    "full_name": full_name,
                    "name": name,
                    "owner": owner,
                    "description": description,
                    "language": language,
                    "stars": stars,
                    "forks": forks,
                    "topics": topics,
                    "license": license_value,
                    "readme": raw_record.get("readme"),
                    "updated_at": updated_at,
                    "pushed_at": pushed_at,
                    "search_text": self._create_search_text(
                        full_name=full_name,
                        description=description,
                        language=language,
                        topics=topics,
                    ),
                }
            )

            # Validate cleaned record
            if not self._validate_cleaned_record(cleaned):
                return None

            return cleaned

        except Exception as e:
            logger.error(f"Error transforming record: {e}")
            return None

    def _validate_cleaned_record(self, record: Dict[str, Any]) -> bool:
        """Validate the cleaned record against required serving constraints."""
        if not record.get("full_name"):
            logger.warning("Missing full_name: %s", record.get("_id"))
            return False
        if not record.get("name"):
            logger.warning("Missing name: %s", record.get("full_name"))
            return False
        if not record.get("language"):
            logger.warning("Missing language after normalization: %s", record.get("full_name"))
            return False
        if not isinstance(record.get("stars"), int):
            logger.warning("Stars is not int: %s", record.get("full_name"))
            return False
        return True

    def _first_non_empty(self, *values: Any) -> Any:
        for value in values:
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            return value
        return None

    def _clean_string(self, value: Any) -> str:
        """Clean and normalize string"""
        if value is None:
            return ""

        value = str(value).strip()
        # Remove excessive whitespace
        value = re.sub(r"\s+", " ", value)
        return value

    def _clean_description(self, description: Any) -> Optional[str]:
        """Clean repository description (nullable)."""
        desc = self._clean_string(description)

        if not desc:
            return None

        # Truncate if too long
        if len(desc) > self.max_desc_length:
            desc = desc[: self.max_desc_length] + "..."

        # Remove common prefixes/patterns
        desc = re.sub(r"^(A |An |The )", "", desc, flags=re.IGNORECASE)

        return desc.strip()

    def _clean_integer(self, value: Any) -> int:
        """Clean and convert to integer"""
        try:
            return max(0, int(value))
        except (ValueError, TypeError):
            return 0

    def _to_iso_timestamp(self, date_value: Any) -> Optional[str]:
        """Parse date-like value and return ISO-8601 string."""
        if not date_value:
            return None

        if isinstance(date_value, datetime):
            return date_value.isoformat()

        try:
            # Try ISO format
            parsed = datetime.fromisoformat(str(date_value).replace("Z", "+00:00"))
            return parsed.isoformat()
        except ValueError:
            return None

    def _clean_topics(self, topics: Any) -> list[str]:
        """Clean and normalize topics"""
        if topics is None:
            return []

        if isinstance(topics, str):
            topics = [p.strip() for p in topics.split(",") if p.strip()]

        if not isinstance(topics, list):
            return []

        cleaned = []
        for topic in topics:
            topic = self._clean_string(topic).lower()
            if topic and len(topic) <= 50:  # Reasonable topic length
                cleaned.append(topic)

        return list(set(cleaned))[:20]  # Max 20 unique topics

    def _normalize_license(self, license_value: Any) -> Optional[str]:
        """Normalize license into string or null."""
        if license_value is None:
            return None

        if isinstance(license_value, dict):
            candidate = self._first_non_empty(
                license_value.get("spdx_id"),
                license_value.get("name"),
                license_value.get("key"),
            )
            cleaned = self._clean_string(candidate)
            return cleaned or None

        cleaned = self._clean_string(license_value)
        return cleaned or None

    def _clean_url(self, url: Any) -> str:
        """Clean and validate URL"""
        url = self._clean_string(url)

        if not url:
            return ""

        # Basic URL validation
        if not url.startswith(("http://", "https://")):
            return ""

        return url

    def _extract_repo_name(self, full_name: str) -> str:
        """Extract repository name from full_name"""
        if not full_name or "/" not in full_name:
            return ""
        return full_name.split("/", 1)[1]

    def _extract_owner(self, full_name: str) -> str:
        """Extract owner from full_name"""
        if not full_name or "/" not in full_name:
            return ""
        return full_name.split("/", 1)[0]

    def _create_search_text(
        self,
        full_name: str,
        description: Optional[str],
        language: str,
        topics: list[str],
    ) -> str:
        """Create searchable text field combining normalized fields."""
        parts = [
            full_name,
            description or "",
            language,
            " ".join(topics or []),
        ]

        search_text = " ".join(filter(None, parts))
        return self._clean_string(search_text).lower()
