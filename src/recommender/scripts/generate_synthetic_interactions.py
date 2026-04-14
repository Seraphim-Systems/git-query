"""Generate synthetic training interactions for the LGBM reranker.

Simulates 20-30 diverse user personas each issuing 10-15 natural-language
queries against the live Qdrant search endpoint.  Each result set is scored
using a heuristic (topic overlap + language match as primary signals, stars
as tiebreaker) and logged to MongoDB with ``source: "synthetic"`` so it can
be filtered separately from real user data.

Run via environment variables:
    MONGODB_URL         – MongoDB connection string
    QDRANT_URL          – Qdrant base URL (e.g. http://localhost:6333)
    QDRANT_API_KEY      – Qdrant API key (optional)
    QDRANT_COLLECTION   – collection name (default: repositories_embeddings)
    EMBEDDING_MODEL     – sentence-transformers model id
    DRY_RUN             – if "1", print interactions without writing to Mongo
"""

from __future__ import annotations

import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Persona definitions
# ---------------------------------------------------------------------------

PERSONAS: List[Dict[str, Any]] = [
    {
        "id": "ml_engineer",
        "languages": ["python"],
        "topics": ["machine-learning", "deep-learning", "pytorch", "tensorflow", "neural-network"],
        "queries": [
            "pytorch training utilities",
            "deep learning model training framework",
            "neural network optimization library",
            "machine learning pipeline tools",
            "gradient descent optimizer python",
            "transformer model fine-tuning",
            "data augmentation for computer vision",
            "hyperparameter tuning library",
            "mlflow experiment tracking",
            "model serving inference",
        ],
    },
    {
        "id": "rust_systems_dev",
        "languages": ["rust"],
        "topics": ["systems-programming", "performance", "memory-safety", "async", "concurrency"],
        "queries": [
            "rust async runtime tokio",
            "high performance networking rust",
            "rust memory safe systems programming",
            "concurrent data structures rust",
            "embedded systems rust no_std",
            "rust cli argument parsing",
            "safe ffi bindings rust",
            "rust serialization serde",
            "low latency rust web server",
            "rust build tools cargo plugins",
        ],
    },
    {
        "id": "js_frontend_dev",
        "languages": ["javascript", "typescript"],
        "topics": ["react", "frontend", "ui", "web", "css"],
        "queries": [
            "react component library",
            "typescript state management",
            "javascript animation library",
            "next.js starter template",
            "tailwind css components",
            "frontend testing library",
            "bundler vite rollup webpack",
            "react hooks utilities",
            "accessibility ui components",
            "responsive design framework",
        ],
    },
    {
        "id": "devops_engineer",
        "languages": ["go", "python", "shell"],
        "topics": ["kubernetes", "docker", "ci-cd", "infrastructure", "monitoring"],
        "queries": [
            "kubernetes operator development",
            "docker container orchestration",
            "ci cd pipeline automation",
            "infrastructure as code terraform",
            "prometheus metrics collection",
            "log aggregation elasticsearch",
            "helm chart templating",
            "service mesh istio envoy",
            "secret management vault",
            "gitops deployment tools",
        ],
    },
    {
        "id": "data_scientist",
        "languages": ["python", "r"],
        "topics": ["data-science", "visualization", "statistics", "pandas", "jupyter"],
        "queries": [
            "pandas dataframe manipulation",
            "data visualization matplotlib plotly",
            "statistical analysis python",
            "jupyter notebook extensions",
            "feature engineering library",
            "time series forecasting",
            "exploratory data analysis tools",
            "data cleaning preprocessing",
            "scikit-learn model evaluation",
            "parquet arrow big data",
        ],
    },
    {
        "id": "backend_go_dev",
        "languages": ["go"],
        "topics": ["api", "grpc", "microservices", "database", "rest"],
        "queries": [
            "golang rest api framework",
            "grpc service implementation go",
            "go database orm library",
            "microservices toolkit golang",
            "go http middleware",
            "golang structured logging",
            "distributed tracing go",
            "go dependency injection",
            "context cancellation patterns go",
            "golang testing mocking",
        ],
    },
    {
        "id": "security_researcher",
        "languages": ["python", "c", "go"],
        "topics": ["security", "cryptography", "vulnerability", "penetration-testing", "fuzzing"],
        "queries": [
            "static analysis security scanner",
            "cryptography library python",
            "fuzzing framework c cpp",
            "vulnerability disclosure tools",
            "tls ssl certificate management",
            "password hashing library",
            "web security testing",
            "network packet analysis",
            "secure coding linter",
            "audit logging library",
        ],
    },
    {
        "id": "mobile_dev",
        "languages": ["swift", "kotlin", "dart"],
        "topics": ["ios", "android", "flutter", "mobile", "ui"],
        "queries": [
            "flutter state management",
            "swift ui components ios",
            "kotlin android architecture",
            "cross platform mobile framework",
            "mobile app navigation library",
            "flutter animations",
            "push notifications mobile",
            "offline storage mobile apps",
            "mobile testing automation",
            "dart async streams",
        ],
    },
    {
        "id": "blockchain_dev",
        "languages": ["solidity", "rust", "typescript"],
        "topics": ["blockchain", "smart-contracts", "defi", "web3", "ethereum"],
        "queries": [
            "solidity smart contract library",
            "ethereum development framework",
            "web3 javascript library",
            "defi protocol implementation",
            "solana program rust",
            "nft minting contract",
            "hardhat truffle testing",
            "decentralized storage ipfs",
            "blockchain indexer subgraph",
            "crypto wallet integration",
        ],
    },
    {
        "id": "game_dev",
        "languages": ["c++", "c#", "lua"],
        "topics": ["game-engine", "graphics", "physics", "2d", "3d"],
        "queries": [
            "game engine ecs entity component",
            "2d game physics engine",
            "opengl vulkan renderer",
            "lua scripting game engine",
            "unity plugin development",
            "procedural generation algorithms",
            "game networking multiplayer",
            "audio engine spatial sound",
            "asset pipeline tools",
            "game state machine library",
        ],
    },
    {
        "id": "data_engineer",
        "languages": ["python", "scala", "java"],
        "topics": ["etl", "pipeline", "kafka", "spark", "data-warehouse"],
        "queries": [
            "apache kafka consumer producer",
            "spark data processing scala",
            "etl pipeline orchestration",
            "dbt data transformation",
            "airflow dag workflow",
            "data lake storage delta",
            "stream processing flink",
            "schema registry avro",
            "data quality testing great expectations",
            "python pipeline prefect",
        ],
    },
    {
        "id": "nlp_researcher",
        "languages": ["python"],
        "topics": ["nlp", "transformers", "llm", "text-processing", "bert"],
        "queries": [
            "huggingface transformers fine-tuning",
            "text classification bert",
            "named entity recognition spacy",
            "sentence embeddings semantic search",
            "llm inference optimization",
            "tokenizer text preprocessing",
            "question answering model",
            "summarization abstractive extractive",
            "multilingual nlp library",
            "retrieval augmented generation",
        ],
    },
    {
        "id": "cloud_architect",
        "languages": ["python", "go", "typescript"],
        "topics": ["aws", "cloud", "serverless", "lambda", "s3"],
        "queries": [
            "aws sdk python boto3 utilities",
            "serverless framework deployment",
            "cloud cost optimization tools",
            "event driven architecture sqs sns",
            "cloud native observability",
            "multi cloud abstraction library",
            "aws cdk infrastructure",
            "lambda cold start optimization",
            "cloud storage s3 compatible",
            "api gateway proxy lambda",
        ],
    },
    {
        "id": "database_admin",
        "languages": ["sql", "python", "go"],
        "topics": ["database", "postgresql", "mysql", "migrations", "performance"],
        "queries": [
            "postgresql performance tuning",
            "database migration tools",
            "sql query optimization",
            "connection pooling library",
            "database backup restore",
            "mongodb aggregation pipeline",
            "redis caching patterns",
            "time series database influxdb",
            "graph database neo4j",
            "database schema versioning",
        ],
    },
    {
        "id": "open_source_contributor",
        "languages": ["python", "javascript", "go", "rust"],
        "topics": ["open-source", "documentation", "testing", "linting", "developer-tools"],
        "queries": [
            "code linting formatting tools",
            "documentation generator sphinx",
            "test coverage reporting",
            "git workflow automation",
            "code review tools",
            "dependency vulnerability scanner",
            "release automation changelog",
            "monorepo management tools",
            "pre-commit hooks framework",
            "developer experience tooling",
        ],
    },
]

# ---------------------------------------------------------------------------
# Scoring heuristic
# ---------------------------------------------------------------------------

_INTERACTION_VALUE_POSITIVE = 3.0
_INTERACTION_VALUE_NEGATIVE = -1.0


def _extract_topics(repo: Dict[str, Any]) -> List[str]:
    raw = repo.get("topics") or []
    if isinstance(raw, list):
        result = []
        for item in raw:
            if isinstance(item, dict):
                result.append((item.get("name") or "").lower())
            else:
                result.append(str(item).lower())
        return [t for t in result if t]
    return []


def score_repo(repo: Dict[str, Any], persona: Dict[str, Any]) -> float:
    """Heuristic relevance score: topic overlap + language match (primary), stars (tiebreaker)."""
    score = 0.0

    repo_topics = set(_extract_topics(repo))
    persona_topics = set(t.lower() for t in persona["topics"])
    if repo_topics and persona_topics:
        overlap = len(repo_topics & persona_topics) / len(persona_topics)
        score += overlap * 10.0

    repo_lang = (repo.get("language") or "").lower()
    persona_langs = set(l.lower() for l in persona["languages"])
    if repo_lang and repo_lang in persona_langs:
        score += 5.0

    stars = repo.get("stars") or repo.get("stargazers_count") or 0
    try:
        stars = int(stars)
    except (TypeError, ValueError):
        stars = 0
    import math
    score += math.log1p(stars) * 0.1

    return score


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------


class SyntheticInteractionGenerator:
    def __init__(
        self,
        mongodb_url: str,
        qdrant_url: str,
        qdrant_collection: str = "repositories_embeddings",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        qdrant_api_key: Optional[str] = None,
        dry_run: bool = False,
    ):
        self.mongodb_url = mongodb_url
        self.qdrant_url = qdrant_url.rstrip("/")
        self.qdrant_collection = qdrant_collection
        self.embedding_model = embedding_model
        self.qdrant_api_key = qdrant_api_key
        self.dry_run = dry_run
        self._embedder = None
        self._mongo_client = None
        self._db = None

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: %s", self.embedding_model)
            self._embedder = SentenceTransformer(self.embedding_model)
        return self._embedder

    def _get_db(self):
        if self._mongo_client is None:
            from pymongo import MongoClient
            self._mongo_client = MongoClient(self.mongodb_url)
            self._db = self._mongo_client.get_default_database()
            if self._db is None:
                db_name = self.mongodb_url.rstrip("/").rsplit("/", 1)[-1].split("?")[0] or "gitquery"
                self._db = self._mongo_client[db_name]
        return self._db

    def embed(self, text: str) -> List[float]:
        return self._get_embedder().encode(text).tolist()

    def search_qdrant(self, vector: List[float], limit: int = 30) -> List[Dict[str, Any]]:
        import requests
        headers = {"Content-Type": "application/json"}
        if self.qdrant_api_key:
            headers["api-key"] = self.qdrant_api_key

        url = f"{self.qdrant_url}/collections/{self.qdrant_collection}/points/query"
        body = {"query": vector, "limit": limit, "with_payload": True}
        resp = requests.post(url, json=body, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data.get("result", {}).get("points") or data.get("result") or []

    def fetch_repo_metadata(self, repo_ids: List[str]) -> Dict[str, Any]:
        db = self._get_db()
        docs = {}
        cursor = db["repositories"].find(
            {"$or": [{"_id": {"$in": repo_ids}}, {"repo_id": {"$in": repo_ids}}]}
        )
        for doc in cursor:
            key = doc.get("repo_id") or str(doc.get("_id", ""))
            docs[key] = doc
        return docs

    def write_interactions(self, interactions: List[Dict[str, Any]]) -> int:
        if self.dry_run:
            for i in interactions:
                logger.info("[DRY RUN] %s", i)
            return len(interactions)
        db = self._get_db()
        result = db["user_interactions"].insert_many(interactions)
        return len(result.inserted_ids)

    def generate_for_persona(self, persona: Dict[str, Any]) -> List[Dict[str, Any]]:
        interactions = []
        persona_id = f"synthetic_{persona['id']}"

        for query in persona["queries"]:
            logger.info("  Query: %r", query)
            try:
                vector = self.embed(query)
                hits = self.search_qdrant(vector, limit=30)
            except Exception as exc:
                logger.warning("Search failed for query %r: %s", query, exc)
                continue

            if not hits:
                logger.warning("No results for query %r", query)
                continue

            repo_ids = [str(h.get("id", "")) for h in hits if h.get("id")]
            metadata_map = self.fetch_repo_metadata(repo_ids)

            scored = []
            for hit in hits:
                rid = str(hit.get("id", ""))
                meta = metadata_map.get(rid) or hit.get("payload") or {}
                s = score_repo(meta, persona)
                scored.append((rid, s, meta))

            scored.sort(key=lambda x: x[1], reverse=True)

            now = datetime.now(timezone.utc)
            for rank, (rid, s, meta) in enumerate(scored, start=1):
                if not rid:
                    continue

                threshold = len(scored) * 0.4
                if rank <= threshold:
                    interaction_type = "click"
                    interaction_value = _INTERACTION_VALUE_POSITIVE
                elif rank <= len(scored) * 0.7:
                    interaction_type = "view"
                    interaction_value = 0.1
                else:
                    interaction_type = "dismiss"
                    interaction_value = _INTERACTION_VALUE_NEGATIVE

                interactions.append(
                    {
                        "interaction_id": str(uuid.uuid4()),
                        "user_id": persona_id,
                        "query": query,
                        "repo_id": rid,
                        "interaction_type": interaction_type,
                        "interaction_value": interaction_value,
                        "position_in_results": rank,
                        "variant": "synthetic",
                        "timestamp": now,
                        "metadata": {
                            "source": "synthetic",
                            "persona": persona["id"],
                            "heuristic_score": round(s, 4),
                        },
                    }
                )

        return interactions

    def run(self, personas: Optional[List[Dict[str, Any]]] = None) -> int:
        if personas is None:
            personas = PERSONAS

        total = 0
        for persona in personas:
            logger.info("Generating interactions for persona: %s", persona["id"])
            interactions = self.generate_for_persona(persona)
            if interactions:
                written = self.write_interactions(interactions)
                logger.info("  Wrote %d interactions", written)
                total += written

        logger.info("Done. Total interactions written: %d", total)
        return total


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    mongodb_url = os.environ.get("MONGODB_URL", "mongodb://localhost:27017/gitquery")
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    qdrant_collection = os.environ.get("QDRANT_COLLECTION", "repositories_embeddings")
    embedding_model = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    qdrant_api_key = os.environ.get("QDRANT_API_KEY") or None
    dry_run = os.environ.get("DRY_RUN", "0") == "1"

    logger.info("=== Synthetic Interaction Generator ===")
    logger.info("MongoDB:    %s", mongodb_url.split("@")[-1])
    logger.info("Qdrant:     %s / %s", qdrant_url, qdrant_collection)
    logger.info("Model:      %s", embedding_model)
    logger.info("Dry run:    %s", dry_run)
    logger.info("Personas:   %d", len(PERSONAS))

    generator = SyntheticInteractionGenerator(
        mongodb_url=mongodb_url,
        qdrant_url=qdrant_url,
        qdrant_collection=qdrant_collection,
        embedding_model=embedding_model,
        qdrant_api_key=qdrant_api_key,
        dry_run=dry_run,
    )
    generator.run()


if __name__ == "__main__":
    main()
