"""Vectorization pipeline - generates embeddings and indexes in Qdrant"""

import logging
from typing import Dict, Any, List
import hashlib
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from processing.config import settings

logger = logging.getLogger(__name__)


class DataVectorizer:
    """Generates embeddings and indexes them in Qdrant"""

    def __init__(self, qdrant_client: QdrantClient):
        self.qdrant = qdrant_client
        self.collection_name = settings.vector_collection

        # Load embedding model (runs once)
        logger.info("Loading embedding model...")
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        logger.info("✓ Embedding model loaded")

    async def vectorize_and_index(self, record: Dict[str, Any]):
        """
        Generate embedding and index in Qdrant

        Args:
            record: Cleaned repository record
        """
        try:
            # Create text for embedding
            embedding_text = self._create_embedding_text(record)

            # Generate embedding
            embedding = self.model.encode(embedding_text).tolist()

            # Create unique ID for Qdrant
            point_id = self._generate_point_id(record["repo_id"])

            # Create point
            point = PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "repo_id": record["repo_id"],
                    "full_name": record["full_name"],
                    "description": record["description"],
                    "language": record["language"],
                    "stars": record["stars"],
                    "topics": record.get("topics", []),
                    "is_archived": record.get("is_archived", False),
                },
            )

            # Upsert to Qdrant
            self.qdrant.upsert(collection_name=self.collection_name, points=[point])

            logger.debug(f"Indexed {record['full_name']} in Qdrant")

        except Exception as e:
            logger.error(f"Error vectorizing record: {e}")
            raise

    def _create_embedding_text(self, record: Dict[str, Any]) -> str:
        """
        Create text for embedding from record fields

        Combines multiple fields to create rich semantic representation
        """
        parts = [
            f"Repository: {record.get('full_name', '')}",
            f"Description: {record.get('description', '')}",
            f"Language: {record.get('language', '')}",
            f"Topics: {', '.join(record.get('topics', []))}",
        ]

        text = ". ".join(filter(None, parts))
        return text[:512]  # Limit length for embedding model

    def _generate_point_id(self, repo_id: str) -> int:
        """
        Generate deterministic integer ID from repo_id

        Qdrant requires integer IDs, so we hash the string ID
        """
        # Create hash and convert to int
        hash_value = hashlib.md5(repo_id.encode()).hexdigest()
        return int(hash_value[:15], 16)  # Use first 15 hex chars

    async def batch_vectorize(self, records: List[Dict[str, Any]]):
        """Vectorize multiple records in batch"""
        if not records:
            return

        try:
            # Dedupe by repo_id to avoid duplicate points in a single batch
            seen_ids = set()
            deduped = []
            skipped = 0
            for r in records:
                rid = r.get("repo_id") or r.get("_id") or r.get("id")
                if not rid:
                    # skip records without id
                    skipped += 1
                    continue
                if rid in seen_ids:
                    skipped += 1
                    logger.debug(f"Skipping duplicate record in batch: {rid}")
                    continue
                seen_ids.add(rid)
                deduped.append(r)

            if skipped:
                logger.info(f"Skipped {skipped} duplicate/invalid records in batch")

            if not deduped:
                return

            # Generate all embeddings at once (faster)
            texts = [self._create_embedding_text(r) for r in deduped]
            embeddings = self.model.encode(texts).tolist()

            # Create points
            points = []
            for record, embedding in zip(deduped, embeddings):
                point_id = self._generate_point_id(record["repo_id"])

                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "repo_id": record["repo_id"],
                        "full_name": record.get("full_name"),
                        "description": record.get("description"),
                        "language": record.get("language"),
                        "stars": record.get("stars"),
                        "topics": record.get("topics", []),
                    },
                )
                points.append(point)

            # Batch upsert
            self.qdrant.upsert(collection_name=self.collection_name, points=points)

            logger.info(f"Batch indexed {len(points)} records in Qdrant")

        except Exception as e:
            logger.error(f"Error in batch vectorization: {e}")
            raise
