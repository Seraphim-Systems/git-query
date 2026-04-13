"""Abstract base class for reranker adapters."""

import abc
from typing import List


class BaseRerankerAdapter(abc.ABC):
    """Common interface for all reranker backends.

    Implementations must be safe to call from a thread-pool executor
    (no asyncio operations inside score()).
    """

    @abc.abstractmethod
    def score(self, query: str, candidates: list) -> List[float]:
        """Score candidates for a query.

        Args:
            query: Search query string.
            candidates: List of RepositoryResult objects.

        Returns:
            List of float scores, same length and order as candidates.
        """
