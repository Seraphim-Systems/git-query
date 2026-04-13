"""MongoDB data fetcher for training pipelines.

Extracted from UnifiedTrainingPipeline to allow reuse across
embedding, cross-encoder, and LightGBM training pipelines.
"""

import hashlib
import json
import logging
import random
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

# Mirrors PersonalizationService signal weights (services/personalization_service.py)
# Keyed on InteractionType string values to avoid importing the serving layer.
INTERACTION_WEIGHTS: Dict[str, float] = {
    "thumbs_up": 5.0,
    "save": 3.0,
    "click": 1.0,
    "view": 0.1,
    "dismiss": -2.0,
    "thumbs_down": -5.0,
}


class MongoDataFetcher:
    """Fetches repository data from the MongoDB API for training.

    Single source of truth for all data-loading operations used by
    the embedding, cross-encoder, and LightGBM training pipelines.
    """

    def __init__(
        self,
        api_base_url: str,
        api_key: str,
        models_dir: str = "/app/models",
    ):
        self.api_base_url = api_base_url.rstrip("/")
        self.api_key = api_key
        self.models_dir = Path(models_dir)
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _stable_id(self, doc: Dict) -> str:
        """Compute a small stable id for a repository document."""
        if not doc:
            return ""
        if doc.get("_id"):
            return str(doc["_id"])
        for key in ("nameWithOwner", "full_name", "repo_id", "id"):
            if doc.get(key):
                return str(doc[key])
        owner = doc.get("owner") or doc.get("owner_login")
        name = doc.get("name")
        if owner and name:
            return f"{owner}/{name}"
        payload = json.dumps(doc, sort_keys=True, default=str).encode("utf-8")
        return hashlib.md5(payload).hexdigest()

    def _dedupe_repositories(self, repositories: List[Dict]) -> List[Dict]:
        """Remove duplicates by stable id, preserving first occurrence."""
        seen: set = set()
        unique: List[Dict] = []
        for r in repositories:
            sid = self._stable_id(r)
            if not sid:
                unique.append(r)
                continue
            if sid in seen:
                continue
            seen.add(sid)
            unique.append(r)
        return unique

    def _get_total_count(self) -> int:
        """Get total number of repositories from the API."""
        try:
            response = requests.post(
                f"{self.api_base_url}/api/mongodb/query",
                headers=self.headers,
                json={
                    "database": "gitquery",
                    "collection": "repositories",
                    "filter": {},
                    "limit": 1,
                    "skip": 0,
                },
                timeout=10,
            )
            response.raise_for_status()
            result = response.json()
            count = result.get("count", 0)
            if count <= 1:
                logger.warning("API reports only %d repos — verifying...", count)
                test_response = requests.post(
                    f"{self.api_base_url}/api/mongodb/query",
                    headers=self.headers,
                    json={
                        "database": "gitquery",
                        "collection": "repositories",
                        "filter": {},
                        "limit": 1000,
                        "skip": 0,
                    },
                    timeout=30,
                )
                test_response.raise_for_status()
                actual_docs = len(test_response.json().get("documents", []))
                if actual_docs > count:
                    return 10_000_000
            return count
        except Exception as e:
            logger.error("Error getting count: %s", e)
            return 10_000_000

    def _fetch_batch(self, skip: int, limit: int, filters: Optional[Dict] = None) -> List[Dict]:
        """Fetch a single batch of repositories."""
        payload = {
            "database": "gitquery",
            "collection": "repositories",
            "filter": filters or {},
            "limit": limit,
            "skip": skip,
            "sort": {"_id": 1},
        }
        try:
            response = requests.post(
                f"{self.api_base_url}/api/mongodb/query",
                headers=self.headers,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()
            return response.json().get("documents", [])
        except Exception as e:
            logger.error("Error fetching batch (skip=%d): %s", skip, e)
            return []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_repositories(
        self,
        batch_size: int = 100,
        max_repos: Optional[int] = None,
    ) -> List[Dict]:
        """Fetch all repositories from the API."""
        logger.info("Fetching repositories from %s", self.api_base_url)
        total_repos = self._get_total_count()
        if max_repos:
            total_repos = min(total_repos, max_repos)
        if total_repos == 0:
            logger.warning("No repositories found on server")
            return []

        all_repos: List[Dict] = []
        skip = 0
        while skip < total_repos:
            current_batch_size = min(batch_size, total_repos - skip)
            batch = self._fetch_batch(skip=skip, limit=current_batch_size)
            if not batch:
                break
            all_repos.extend(batch)
            skip += batch_size
            time.sleep(0.1)

        logger.info("Fetched %d repositories", len(all_repos))
        return all_repos

    def check_for_new_data(self, repositories: List[Dict]) -> Tuple[bool, List[Dict]]:
        """Check if there is new data compared to the last training run."""
        latest_mapping = self.models_dir / "metadata" / "repo_mapping_latest.json"
        if not latest_mapping.exists():
            logger.info("No previous mapping — first run")
            return True, repositories
        try:
            with open(latest_mapping) as f:
                mapping_list = json.load(f)
            prev_ids = {m.get("repo_id") for m in mapping_list if m.get("repo_id")}
            current_ids = {self._stable_id(r) for r in repositories}
            added = current_ids - prev_ids
            removed = prev_ids - current_ids
            if added or removed:
                logger.info("Detected %d added, %d removed repos", len(added), len(removed))
                return True, repositories
            logger.info("No repository set changes detected")
            return False, []
        except Exception as e:
            logger.warning("Could not read previous mapping: %s", e)
            return True, repositories

    def fetch_interactions(self, days: int = 90) -> Dict[str, float]:
        """Fetch user interactions and return a repo_id → weighted score mapping.

        Queries the ``user_interactions`` collection for the last *days* days and
        aggregates per-repo interaction scores using ``INTERACTION_WEIGHTS``.
        Returns an empty dict (gracefully) if the collection is unreachable.
        """
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
        try:
            response = requests.post(
                f"{self.api_base_url}/api/mongodb/query",
                headers=self.headers,
                json={
                    "database": "gitquery",
                    "collection": "user_interactions",
                    "filter": {"timestamp": {"$gte": cutoff}},
                    "limit": 100000,
                    "skip": 0,
                },
                timeout=30,
            )
            response.raise_for_status()
            docs = response.json().get("documents", [])
            scores: Dict[str, float] = {}
            for doc in docs:
                repo_id = doc.get("repo_id", "")
                weight = INTERACTION_WEIGHTS.get(doc.get("interaction_type", ""), 0.0)
                if repo_id and weight != 0.0:
                    scores[repo_id] = scores.get(repo_id, 0.0) + weight
            logger.info(
                "Fetched %d interactions → %d repos with non-zero scores (last %d days)",
                len(docs),
                len(scores),
                days,
            )
            return scores
        except Exception as e:
            logger.warning("Could not fetch interactions — falling back to star-based labels: %s", e)
            return {}

    def fetch_training_pairs(
        self,
        max_repos: Optional[int] = None,
        top_n_queries: int = 100,
        candidates_per_query: int = 100,
        interaction_days: int = 90,
    ) -> Dict:
        """Fetch repositories and build training pair structures.

        Args:
            max_repos: Maximum number of repositories to fetch (None = all).
            top_n_queries: Number of top queries to retain in build_query_groups.
            candidates_per_query: Number of candidate repos per query in build_query_groups.

        Returns a dict consumed by all trainer interfaces:
          - "queries": list of query strings
          - "positive_repos": list of repo dicts (positive examples)
          - "negative_repos": list of repo dicts (negative examples)
          - "grouped_df": pd.DataFrame for LGBMRanker (from build_query_groups),
            includes an ``interaction_score`` column (0.0 where no data)
          - "dataset_version": deterministic hash string
        """
        import pandas as pd

        from ..lgbm_ranker import build_query_groups

        repositories = self.fetch_repositories(max_repos=max_repos)
        repositories = self._dedupe_repositories(repositories)

        # Build grouped_df used by the LightGBM pipeline
        df = pd.DataFrame(repositories)
        for col in ("name", "description", "language", "stars"):
            if col not in df.columns:
                df[col] = None

        # Join interaction scores — enables dot-product relevance labels in LGBMRanker
        interaction_scores = self.fetch_interactions(days=interaction_days)
        if interaction_scores:
            df["interaction_score"] = df.apply(
                lambda row: interaction_scores.get(self._stable_id(row.to_dict()), 0.0),
                axis=1,
            )
            coverage = (df["interaction_score"] != 0).sum()
            logger.info(
                "Interaction coverage: %d / %d repos (%.1f%%)",
                coverage,
                len(df),
                100 * coverage / max(len(df), 1),
            )
        else:
            df["interaction_score"] = 0.0
            logger.info("No interaction data — training will use star-based labels")

        grouped_df = build_query_groups(df, top_n_queries=top_n_queries, candidates_per_query=candidates_per_query)

        # Build query/positive/negative lists for embedding + cross-encoder pipelines
        queries: List[str] = []
        positive_repos: List[Dict] = []
        negative_repos: List[Dict] = []

        lang_groups: Dict[str, List[Dict]] = {}
        for repo in repositories:
            lang = (repo.get("language") or "unknown").strip().lower()
            lang_groups.setdefault(lang, []).append(repo)

        rng = random.Random(42)
        for lang, repos in lang_groups.items():
            if len(repos) < 2:
                continue
            sorted_repos = sorted(
                repos,
                key=lambda r: r.get("stargazers_count", 0) or r.get("stars", 0),
                reverse=True,
            )
            for repo in sorted_repos[: max(1, len(sorted_repos) // 2)]:
                queries.append(lang)
                positive_repos.append(repo)
                other_langs = [ll for ll in lang_groups if ll != lang and lang_groups[ll]]
                if other_langs:
                    neg_lang = rng.choice(other_langs)
                    negative_repos.append(rng.choice(lang_groups[neg_lang]))
                else:
                    negative_repos.append(sorted_repos[-1])

        # Deterministic dataset version based on repo IDs
        all_ids = sorted([self._stable_id(r) for r in repositories])
        version_hash = hashlib.sha256("\n".join(all_ids).encode()).hexdigest()[:12]
        dataset_version = f"{datetime.now().strftime('%Y%m%d')}-{version_hash}"

        return {
            "queries": queries,
            "positive_repos": positive_repos,
            "negative_repos": negative_repos,
            "grouped_df": grouped_df,
            "dataset_version": dataset_version,
        }
