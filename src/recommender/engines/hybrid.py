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

        # vector_search handles the run_in_executor internally.
        results = await db_manager.vector_search(
            query_vector=query_embedding,
            top_k=settings.hybrid_search_top_k,
        )

        out = []
        for r in results:
            payload = r.get("payload") or {}
            # Prefer the string repo_id stored in the Qdrant payload over the
            # point UUID so downstream RRF and MongoDB lookups use stable IDs.
            string_id = (
                payload.get("repo_id")
                or payload.get("_orig_id")
                or str(r["repo_id"])
            )
            out.append({
                "repo_id": string_id,
                "score": r["score"],
                "source": "semantic",
                "payload": payload,
            })

        # Batch-fetch full metadata for repos that only have minimal payload.
        # This enriches results even when Qdrant payloads lack name/stars/etc.
        ids_needing_enrichment = [
            r["repo_id"] for r in out
            if "/" in r["repo_id"] and not r["payload"].get("stars")
        ]
        if ids_needing_enrichment:
            meta_map = await db_manager.get_repositories_by_repo_ids(ids_needing_enrichment)
            if meta_map:
                for r in out:
                    if r["repo_id"] in meta_map and not r["payload"].get("stars"):
                        r["payload"] = {**r["payload"], **meta_map[r["repo_id"]]}

        return out

    async def _keyword_search(
        self, request: RecommendationRequest
    ) -> List[Dict[str, Any]]:
        """Search using keywords."""
        query_filter = {}

        if request.query:
            # Use MongoDB Text Search ($text) for O(log N) performance
            # Requires a text index on name, description, and topics
            query_filter["$text"] = {"$search": request.query}

        repos = await db_manager.search_repositories(
            query_filter=query_filter,
            limit=settings.hybrid_search_top_k,
        )

        return [
            {
                "repo_id": repo.get("repo_id", str(repo.get("_id"))),
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
            sources.setdefault(repo_id, set()).add("semantic")

            # Fallback metadata from Qdrant payload
            if "payload" in result and repo_id not in repo_data:
                repo_data[repo_id] = result["payload"]

        # Process keyword results
        for rank, result in enumerate(keyword_results, start=1):
            repo_id = result["repo_id"]
            scores[repo_id] = scores.get(repo_id, 0) + 1 / (self.k + rank)
            sources.setdefault(repo_id, set()).add("keyword")
            if "repo_data" in result:
                # Keyword data is usually richer, so it takes precedence
                repo_data[repo_id] = result["repo_data"]

        # Sort by fused score
        sorted_repos = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        # Convert to RepositoryResult — use whatever data we have; a minimal
        # payload (repo_id only) is better than skipping the result entirely.
        results = []
        for repo_id, score in sorted_repos:
            data = repo_data.get(repo_id) or {}
            # Derive a human-readable name from the repo_id string when full
            # metadata is absent (e.g. Qdrant payload not yet populated).
            display_name = (
                data.get("name")
                or data.get("full_name")
                or (repo_id.split("/")[-1] if "/" in str(repo_id) else "")
            )
            results.append(
                RepositoryResult(
                    repo_id=repo_id,
                    name=display_name,
                    full_name=data.get("full_name", repo_id if "/" in str(repo_id) else ""),
                    description=data.get("description"),
                    language=data.get("language"),
                    stars=data.get("stars", data.get("stargazers_count", 0)),
                    forks=data.get("forks", data.get("forks_count", 0)),
                    url=data.get("url", data.get("html_url", "")),
                    license=data.get("license"),
                    last_updated=data.get("last_updated", data.get("updated_at")),
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

