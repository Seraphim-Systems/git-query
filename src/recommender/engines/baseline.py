"""Baseline recommendation engine - keyword search only."""

from typing import List, Dict, Any
from ..models import RecommendationRequest, RepositoryResult
from ..database import db_manager
from .base import RecommendationEngine


class BaselineEngine(RecommendationEngine):
    """
    Simple baseline engine using keyword search only.
    Good for establishing baseline metrics for A/B testing.
    """

    def __init__(self):
        super().__init__(name="baseline", version="1.0.0")

    async def recommend(
        self, request: RecommendationRequest
    ) -> List[RepositoryResult]:
        """Generate recommendations using keyword search."""
        # Build filter
        query_filter = {}

        # Text search on name and description
        if request.query:
            query_filter["$or"] = [
                {"name": {"$regex": request.query, "$options": "i"}},
                {"description": {"$regex": request.query, "$options": "i"}},
                {"topics": {"$regex": request.query, "$options": "i"}},
            ]

        # Apply hard filters
        if request.language:
            query_filter["language"] = request.language
        if request.min_stars:
            query_filter["stars"] = {"$gte": request.min_stars}
        if request.license:
            query_filter["license"] = request.license

        # Search repositories
        repos = await db_manager.search_repositories(
            query_filter=query_filter,
            limit=request.top_k,
        )

        # Convert to results
        results = []
        for idx, repo in enumerate(repos):
            results.append(
                RepositoryResult(
                    repo_id=repo.get("repo_id", repo.get("_id")),
                    name=repo.get("name", ""),
                    full_name=repo.get("full_name", ""),
                    description=repo.get("description"),
                    language=repo.get("language"),
                    stars=repo.get("stars", 0),
                    forks=repo.get("forks", 0),
                    url=repo.get("url", ""),
                    license=repo.get("license"),
                    last_updated=repo.get("last_updated"),
                    score=1.0 / (idx + 1),  # Simple ranking score
                    rank=idx + 1,
                    explanation={"method": "keyword_search"},
                )
            )

        return results

    async def explain(
        self, repo_id: str, request: RecommendationRequest
    ) -> Dict[str, Any]:
        """Explain baseline ranking."""
        return {
            "engine": self.name,
            "method": "keyword_search",
            "query": request.query,
            "message": "Repository matched keyword search criteria",
        }

