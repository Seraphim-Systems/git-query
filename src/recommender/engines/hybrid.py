"""Hybrid retrieval engine - combines embeddings + keyword search."""

from typing import List, Dict, Any, Set
from ..models import RecommendationRequest, RepositoryResult
from ..database import db_manager
from ..config import settings
from .base import RecommendationEngine
import asyncio


class HybridRetrievalEngine(RecommendationEngine):
    """
    Hybrid retrieval combining:
    1. Semantic search via embeddings (for meaning)
    2. Keyword search (for exact terms)
    3. Score fusion (RRF - Reciprocal Rank Fusion)
    """

    def __init__(self, embedding_service=None, reranker_service=None):
        super().__init__(name="hybrid", version="1.0.0")
        self.embedding_service = embedding_service
        self.reranker_service = reranker_service
        self.k = 60  # RRF constant

    async def recommend(
        self, request: RecommendationRequest
    ) -> List[RepositoryResult]:
        """Generate recommendations using hybrid retrieval."""

        # Step 1: Parallel retrieval from both sources
        semantic_task = self._semantic_search(request)
        keyword_task = self._keyword_search(request)

        semantic_results, keyword_results = await asyncio.gather(
            semantic_task, keyword_task
        )

        # Step 2: Fuse results using Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(
            semantic_results, keyword_results
        )

        # Step 3: Apply hard filters
        filtered_results = self._apply_filters(fused_results, request)

        # Step 4: Rerank top candidates if reranker available
        if self.reranker_service and len(filtered_results) > 0:
            top_candidates = filtered_results[:settings.hybrid_search_top_k]
            reranked = await self.reranker_service.rerank(
                query=request.query,
                candidates=top_candidates,
                top_k=settings.rerank_top_k,
            )
            filtered_results = reranked + filtered_results[settings.hybrid_search_top_k:]

        # Step 5: Return top K
        final_results = filtered_results[:request.top_k]

        # Add rank
        for idx, result in enumerate(final_results):
            result.rank = idx + 1

        return final_results

    async def _semantic_search(
        self, request: RecommendationRequest
    ) -> List[Dict[str, Any]]:
        """Search using semantic embeddings."""
        if not self.embedding_service:
            return []

        # Get query embedding
        query_embedding = await self.embedding_service.embed_text(request.query)

        # Search in Qdrant
        results = db_manager.vector_search(
            query_vector=query_embedding,
            top_k=settings.hybrid_search_top_k,
        )

        return [{"repo_id": r["repo_id"], "score": r["score"], "source": "semantic"}
                for r in results]

    async def _keyword_search(
        self, request: RecommendationRequest
    ) -> List[Dict[str, Any]]:
        """Search using keywords."""
        query_filter = {}

        if request.query:
            query_filter["$or"] = [
                {"name": {"$regex": request.query, "$options": "i"}},
                {"description": {"$regex": request.query, "$options": "i"}},
                {"topics": {"$in": [request.query]}},
            ]

        repos = await db_manager.search_repositories(
            query_filter=query_filter,
            limit=settings.hybrid_search_top_k,
        )

        return [
            {
                "repo_id": repo.get("repo_id", repo.get("_id")),
                "repo_data": repo,
                "score": 1.0,
                "source": "keyword",
            }
            for repo in repos
        ]

    def _reciprocal_rank_fusion(
        self, semantic_results: List[Dict], keyword_results: List[Dict]
    ) -> List[RepositoryResult]:
        """Fuse results using Reciprocal Rank Fusion."""
        scores: Dict[str, float] = {}
        repo_data: Dict[str, Dict] = {}
        sources: Dict[str, Set[str]] = {}

        # Process semantic results
        for rank, result in enumerate(semantic_results, start=1):
            repo_id = result["repo_id"]
            scores[repo_id] = scores.get(repo_id, 0) + 1 / (self.k + rank)
            sources[repo_id] = sources.get(repo_id, set())
            sources[repo_id].add("semantic")

        # Process keyword results
        for rank, result in enumerate(keyword_results, start=1):
            repo_id = result["repo_id"]
            scores[repo_id] = scores.get(repo_id, 0) + 1 / (self.k + rank)
            sources[repo_id] = sources.get(repo_id, set())
            sources[repo_id].add("keyword")
            if "repo_data" in result:
                repo_data[repo_id] = result["repo_data"]

        # Sort by fused score
        sorted_repos = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Convert to RepositoryResult
        results = []
        for repo_id, score in sorted_repos:
            data = repo_data.get(repo_id, {})
            if not data:
                continue  # Skip if we don't have repo data

            results.append(
                RepositoryResult(
                    repo_id=repo_id,
                    name=data.get("name", ""),
                    full_name=data.get("full_name", ""),
                    description=data.get("description"),
                    language=data.get("language"),
                    stars=data.get("stars", 0),
                    forks=data.get("forks", 0),
                    url=data.get("url", ""),
                    license=data.get("license"),
                    last_updated=data.get("last_updated"),
                    score=score,
                    rank=0,  # Will be set later
                    explanation={
                        "method": "hybrid_rrf",
                        "sources": list(sources.get(repo_id, [])),
                        "rrf_score": score,
                    },
                )
            )

        return results

    def _apply_filters(
        self, results: List[RepositoryResult], request: RecommendationRequest
    ) -> List[RepositoryResult]:
        """Apply hard constraints - filters are never violated."""
        filtered = []

        for result in results:
            # Language filter
            if request.language and result.language != request.language:
                continue

            # Min stars filter
            if request.min_stars and result.stars < request.min_stars:
                continue

            # License filter
            if request.license and result.license != request.license:
                continue

            filtered.append(result)

        return filtered

    async def explain(
        self, repo_id: str, request: RecommendationRequest
    ) -> Dict[str, Any]:
        """Explain hybrid ranking."""
        return {
            "engine": self.name,
            "method": "hybrid_retrieval_rrf",
            "query": request.query,
            "message": "Repository retrieved via hybrid search (semantic + keyword) and fused with RRF",
        }

