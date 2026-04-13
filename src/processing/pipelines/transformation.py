"""Data transformation pipeline - normalizes raw repository documents."""

import logging
import math
import re
from typing import Dict, Any, Optional
from datetime import datetime, timezone

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
            # Build a curated cleaned schema rather than copying all raw fields.
            cleaned: Dict[str, Any] = {}

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
                self._extract_owner_login(raw_record.get("owner")),
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

            watchers = self._clean_integer(
                self._first_non_empty(raw_record.get("watchers"), raw_record.get("watcherCount"), 0)
            )
            issues = self._clean_integer(
                self._first_non_empty(raw_record.get("issues"), raw_record.get("openIssues"), 0)
            )
            pull_requests = self._clean_integer(
                self._first_non_empty(raw_record.get("pullRequests"), raw_record.get("pull_requests"), 0)
            )

            is_fork = bool(
                self._first_non_empty(raw_record.get("is_fork"), raw_record.get("isFork"), False)
            )
            is_archived = bool(
                self._first_non_empty(raw_record.get("is_archived"), raw_record.get("isArchived"), False)
            )
            forking_allowed = bool(self._first_non_empty(raw_record.get("forkingAllowed"), True))

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
                    "watchers": watchers,
                    "issues": issues,
                    "pull_requests": pull_requests,
                    "topics": topics,
                    "license": license_value,
                    "readme": raw_record.get("readme"),
                    "is_fork": is_fork,
                    "is_archived": is_archived,
                    "forking_allowed": forking_allowed,
                    "updated_at": updated_at,
                    "pushed_at": pushed_at,
                    "created_at": self._to_iso_timestamp(
                        self._first_non_empty(raw_record.get("created_at"), raw_record.get("createdAt"))
                    ),
                    "search_text": self._create_search_text(
                        full_name=full_name,
                        description=description,
                        language=language,
                        topics=topics,
                    ),
                }
            )

            # Model/EDA-ready engineered features stored per cleaned repository.
            cleaned.update(self._engineered_features(cleaned))

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

    def _parse_iso_datetime(self, value: Any) -> Optional[datetime]:
        """Best-effort parser for ISO timestamps used by recency features."""
        if not value:
            return None
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed
        except ValueError:
            return None

    def _days_since(self, iso_value: Any) -> Optional[int]:
        parsed = self._parse_iso_datetime(iso_value)
        if parsed is None:
            return None
        return max(0, int((datetime.now(timezone.utc) - parsed).days))

    def _is_permissive_license(self, license_value: Optional[str]) -> int:
        if not license_value:
            return 0
        permissive = {
            "mit",
            "apache-2.0",
            "bsd-2-clause",
            "bsd-3-clause",
            "isc",
            "unlicense",
        }
        normalized = self._clean_string(license_value).lower()
        return int(normalized in permissive)

    def _engineered_features(self, cleaned: Dict[str, Any]) -> Dict[str, Any]:
        stars = self._clean_integer(cleaned.get("stars"))
        forks = self._clean_integer(cleaned.get("forks"))

        topics = cleaned.get("topics") if isinstance(cleaned.get("topics"), list) else []
        description = cleaned.get("description") or ""
        readme = cleaned.get("readme") or ""

        pushed_days = self._days_since(cleaned.get("pushed_at"))
        updated_days = self._days_since(cleaned.get("updated_at"))
        days_since_update = pushed_days if pushed_days is not None else updated_days

        return {
            "stars_log": round(math.log1p(stars), 6),
            "forks_log": round(math.log1p(forks), 6),
            "fork_star_ratio": round(float(forks) / float(stars + 1), 6),
            "has_readme": int(bool(str(readme).strip())),
            "readme_length": len(str(readme)),
            "description_length": len(str(description)),
            "num_topics": len(topics),
            "has_license": int(bool(cleaned.get("license"))),
            "is_permissive": self._is_permissive_license(cleaned.get("license")),
            "days_since_update": days_since_update,
            "is_stale": int(days_since_update is not None and days_since_update > 730),
        }

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

    def _extract_owner_login(self, owner_value: Any) -> Optional[str]:
        """Extract owner login when owner is a nested object."""
        if isinstance(owner_value, dict):
            candidate = self._first_non_empty(
                owner_value.get("login"),
                owner_value.get("name"),
                owner_value.get("id"),
            )
            return self._clean_string(candidate)
        if owner_value is None:
            return None
        return self._clean_string(owner_value)

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
