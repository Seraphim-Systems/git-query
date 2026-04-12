"""MCP tools for repository data inspection and explanation."""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import httpx
from pymongo.errors import PyMongoError

from src.db.clients import get_mongo_client
from src.db.config import db_clients

logger = logging.getLogger(__name__)

_COLLECTION_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+$")
_FULL_NAME_RE = re.compile(r"^(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)$")
_GITHUB_REPO_URL_RE = re.compile(
    r"https?://github\.com/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)(?:/.*)?$",
    re.IGNORECASE,
)
_DISALLOWED_FILTER_KEYS = {"$where", "$function", "$accumulator"}


def _serialize_value(value: Any) -> Any:
    """Convert Mongo-native values into JSON-safe Python types."""
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "isoformat") and callable(value.isoformat):
        return value.isoformat()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _validate_safe_filter(value: Any) -> bool:
    """Reject filter operators that can execute server-side code."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in _DISALLOWED_FILTER_KEYS:
                return False
            if not _validate_safe_filter(item):
                return False
        return True
    if isinstance(value, list):
        return all(_validate_safe_filter(item) for item in value)
    return True


def _get_db(database: Optional[str] = None):
    """Resolve Mongo DB from runtime clients or lazy fallback config client."""
    db_name = (database or db_clients.config.mongodb_db or "gitquery").strip()
    runtime_client = get_mongo_client()
    if runtime_client is not None:
        return runtime_client[db_name]
    client = db_clients.mongodb
    return client[db_name]


def _extract_full_name(
    repo_url: Optional[str], full_name: Optional[str]
) -> Optional[str]:
    if full_name:
        name = full_name.strip()
        if _FULL_NAME_RE.match(name):
            return name
        return None
    if not repo_url:
        return None
    match = _GITHUB_REPO_URL_RE.match(repo_url.strip())
    if not match:
        return None
    return f"{match.group('owner')}/{match.group('repo')}"


def _build_repo_summary(
    repo_data: dict[str, Any], readme_excerpt: Optional[str]
) -> dict[str, Any]:
    """Create a concise explanation payload from metadata plus README text."""
    language = repo_data.get("language")
    stars = repo_data.get("stargazers_count") or repo_data.get("stars")
    forks = repo_data.get("forks_count") or repo_data.get("forks")
    open_issues = repo_data.get("open_issues_count")
    topics = repo_data.get("topics") or []

    highlights = []
    description = repo_data.get("description")
    if description:
        highlights.append(f"Purpose: {description}")
    if language:
        highlights.append(f"Primary language: {language}")
    if stars is not None:
        highlights.append(f"Community traction: {stars} stars")
    if forks is not None:
        highlights.append(f"Forks: {forks}")
    if open_issues is not None:
        highlights.append(f"Open issues: {open_issues}")
    if topics:
        highlights.append(f"Topics: {', '.join(topics[:8])}")
    if readme_excerpt:
        highlights.append(f"README excerpt: {readme_excerpt}")

    return {
        "repository": {
            "full_name": repo_data.get("full_name"),
            "url": repo_data.get("html_url") or repo_data.get("url"),
            "description": description,
            "language": language,
            "stars": stars,
            "forks": forks,
            "topics": topics,
            "updated_at": repo_data.get("updated_at"),
        },
        "highlights": highlights,
    }


async def query_repository_data(
    collection: str,
    filters: Optional[dict[str, Any]] = None,
    projection: Optional[list[str]] = None,
    sort_by: Optional[str] = None,
    sort_order: str = "desc",
    limit: int = 20,
    skip: int = 0,
    database: Optional[str] = None,
) -> dict[str, Any]:
    """
    Run a read-only query against MongoDB and return JSON-safe documents.

    Args:
        collection: MongoDB collection name.
        filter: MongoDB filter object.
        projection: List of fields to include in each result document.
        sort_by: Optional field name used for sorting.
        sort_order: "asc" or "desc".
        limit: Max number of returned documents (1-100).
        skip: Number of matched documents to skip.
        database: Optional DB name. Defaults to configured Mongo DB.
    """
    if not collection or not _COLLECTION_NAME_RE.match(collection):
        return {
            "error": "Invalid collection name. Use letters, numbers, underscores, or hyphens only."
        }

    query_filter = filters or {}
    if not isinstance(query_filter, dict):
        return {"error": "Filter must be a JSON object."}

    if not _validate_safe_filter(query_filter):
        return {
            "error": "Filter contains disallowed operators ($where, $function, or $accumulator)."
        }

    projection_doc = None
    if projection is not None:
        if not isinstance(projection, list) or not all(
            isinstance(field, str) and field for field in projection
        ):
            return {"error": "Projection must be a list of non-empty field names."}
        projection_doc = {field: 1 for field in projection}

    normalized_limit = max(1, min(int(limit), 100))
    normalized_skip = max(0, int(skip))
    normalized_order = 1 if str(sort_order).lower() == "asc" else -1

    try:
        db = _get_db(database)
        cursor = db[collection].find(query_filter, projection_doc)
        if sort_by:
            cursor = cursor.sort(sort_by, normalized_order)
        cursor = cursor.skip(normalized_skip).limit(normalized_limit)
        documents = [_serialize_value(doc) for doc in list(cursor)]
        total_matches = db[collection].count_documents(query_filter)

        return {
            "database": db.name,
            "collection": collection,
            "count": len(documents),
            "total_matches": total_matches,
            "limit": normalized_limit,
            "skip": normalized_skip,
            "sort": (
                {
                    "field": sort_by,
                    "order": "asc" if normalized_order == 1 else "desc",
                }
                if sort_by
                else None
            ),
            "documents": documents,
        }
    except (PyMongoError, TypeError, ValueError) as e:
        logger.error("query_repository_data failed: %s", e, exc_info=True)
        return {"error": "Failed to query database", "detail": str(e)}


async def explain_repository(
    repo_url: Optional[str] = None,
    full_name: Optional[str] = None,
    include_database_context: bool = True,
    include_readme_excerpt: bool = True,
) -> dict[str, Any]:
    """
    Explain what a repository is about using DB metadata and GitHub data.

    If neither `repo_url` nor `full_name` is provided, this returns an
    explanation of the current Git-Query project based on local README context.
    """
    resolved_full_name = _extract_full_name(repo_url, full_name)

    if not resolved_full_name:
        readme_path = Path(__file__).resolve().parents[3] / "README.md"
        try:
            content = readme_path.read_text(encoding="utf-8")
            excerpt_lines = [
                line.strip() for line in content.splitlines() if line.strip()
            ]
            excerpt = " ".join(excerpt_lines[:10])[:700]
        except OSError:
            excerpt = "Git-Query is a modular GitHub repository recommendation system with MCP, recommender, and training services."

        return {
            "target": "git-query",
            "summary": "Local project explanation generated from repository README.",
            "highlights": [excerpt],
            "data_sources": {
                "local_readme": True,
                "mongodb": False,
                "github_api": False,
            },
        }

    db_payload: Optional[dict[str, Any]] = None
    if include_database_context:
        try:
            db = _get_db(None)
            db_doc = db["repositories"].find_one(
                {
                    "$or": [
                        {"full_name": resolved_full_name},
                        {"repo_id": resolved_full_name},
                    ]
                }
            )
            if db_doc:
                db_payload = _serialize_value(db_doc)
        except PyMongoError as e:
            logger.warning("Database lookup failed for %s: %s", resolved_full_name, e)

    github_repo: dict[str, Any] = {}
    readme_excerpt: Optional[str] = None

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "git-query-mcp/1.0",
    }
    async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
        try:
            repo_resp = await client.get(
                f"https://api.github.com/repos/{resolved_full_name}"
            )
            if repo_resp.status_code == 200:
                github_repo = repo_resp.json()
        except httpx.HTTPError as e:
            logger.warning("GitHub repo fetch failed for %s: %s", resolved_full_name, e)

        if include_readme_excerpt:
            try:
                readme_resp = await client.get(
                    f"https://api.github.com/repos/{resolved_full_name}/readme",
                    headers={
                        **headers,
                        "Accept": "application/vnd.github.raw+json",
                    },
                )
                if readme_resp.status_code == 200:
                    readme_text = readme_resp.text
                    lines = [
                        line.strip()
                        for line in readme_text.splitlines()
                        if line.strip()
                    ]
                    readme_excerpt = " ".join(lines[:5])[:500]
            except httpx.HTTPError as e:
                logger.warning(
                    "GitHub README fetch failed for %s: %s", resolved_full_name, e
                )

    merged = {}
    merged.update(db_payload or {})
    merged.update(github_repo or {})
    merged.setdefault("full_name", resolved_full_name)

    explanation = _build_repo_summary(merged, readme_excerpt)
    return {
        "target": resolved_full_name,
        "summary": "Repository explanation generated from available metadata sources.",
        "data_sources": {
            "mongodb": bool(db_payload),
            "github_api": bool(github_repo),
            "github_readme": bool(readme_excerpt),
        },
        "database_match": db_payload,
        **explanation,
    }


TOOL_DEFINITIONS = [
    {
        "name": "query_repository_data",
        "description": (
            "Run a read-only MongoDB query against repository data. "
            "Useful for retrieving raw repository records, activity data, and user-related documents."
        ),
        "parameters": [
            {
                "name": "collection",
                "type": "string",
                "description": "MongoDB collection name (e.g. repositories, users)",
                "required": True,
            },
            {
                "name": "filters",
                "type": "object",
                "description": "MongoDB filter object",
                "required": False,
                "default": {},
            },
            {
                "name": "projection",
                "type": "array",
                "description": "List of fields to include in results",
                "required": False,
            },
            {
                "name": "sort_by",
                "type": "string",
                "description": "Field name to sort by",
                "required": False,
            },
            {
                "name": "sort_order",
                "type": "string",
                "description": "Sort order: asc or desc (default desc)",
                "required": False,
                "default": "desc",
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "Maximum number of documents (1-100)",
                "required": False,
                "default": 20,
            },
            {
                "name": "skip",
                "type": "integer",
                "description": "Number of documents to skip",
                "required": False,
                "default": 0,
            },
            {
                "name": "database",
                "type": "string",
                "description": "Optional MongoDB database name",
                "required": False,
            },
        ],
    },
    {
        "name": "explain_repository",
        "description": (
            "Explain what a repository does using available metadata from MongoDB and GitHub. "
            "Provide either a GitHub URL or owner/repo full name."
        ),
        "parameters": [
            {
                "name": "repo_url",
                "type": "string",
                "description": "GitHub repository URL (e.g. https://github.com/owner/repo)",
                "required": False,
            },
            {
                "name": "full_name",
                "type": "string",
                "description": "Repository full name (owner/repo)",
                "required": False,
            },
            {
                "name": "include_database_context",
                "type": "boolean",
                "description": "Include repository metadata from MongoDB when available",
                "required": False,
                "default": True,
            },
            {
                "name": "include_readme_excerpt",
                "type": "boolean",
                "description": "Fetch and include a short excerpt from the GitHub README",
                "required": False,
                "default": True,
            },
        ],
    },
]
