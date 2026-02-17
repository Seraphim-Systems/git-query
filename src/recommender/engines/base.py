"""Base recommendation engine interface (SOLID: Interface Segregation Principle)."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from ..models import RecommendationRequest, RepositoryResult


class RecommendationEngine(ABC):
    """
    Abstract base class for all recommendation engines.

    Following SOLID principles:
    - Single Responsibility: Each engine has one recommendation strategy
    - Open/Closed: Open for extension (new engines), closed for modification
    - Liskov Substitution: All engines can be used interchangeably
    - Interface Segregation: Minimal interface
    - Dependency Inversion: Depends on abstractions, not concretions
    """

    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version

    @abstractmethod
    async def recommend(
        self, request: RecommendationRequest
    ) -> List[RepositoryResult]:
        """
        Generate repository recommendations for a given request.

        Args:
            request: The recommendation request with query and filters

        Returns:
            List of repository results sorted by relevance
        """
        pass

    @abstractmethod
    async def explain(
        self, repo_id: str, request: RecommendationRequest
    ) -> Dict[str, Any]:
        """
        Explain why a repository was recommended.

        Args:
            repo_id: The repository ID
            request: The original request

        Returns:
            Dictionary with explanation details
        """
        pass

    def get_metadata(self) -> Dict[str, str]:
        """Get engine metadata."""
        return {"name": self.name, "version": self.version}

