"""Repos router — public lookup of repository details by repo_id.

Accepts a list of repo_ids (== MongoDB `_id` values, which are the
`nameWithOwner` strings written by kaggle_to_mongo.py) and returns
normalised repo objects ready for display.  No client-side API key is
needed; the service uses the internal MongoDB client directly.
"""

from fastapi import APIRouter, Body, HTTPException, Request
from typing import List, Dict, Any

from src.db.clients import get_mongo_client

router = APIRouter(prefix="/repos", tags=["Repos"])


def _normalize(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Map raw Kaggle/MongoDB document fields to the display shape."""
    full_name: str = (
        doc.get("nameWithOwner")
        or doc.get("full_name")
        or doc.get("repo_id")
        or str(doc.get("_id", ""))
    )

    # owner: stored as {login: "..."} in Kaggle data
    owner_raw = doc.get("owner", "")
    if isinstance(owner_raw, dict):
        owner = owner_raw.get("login", "Unknown")
    else:
        owner = owner_raw or (full_name.split("/")[0] if "/" in full_name else "Unknown")

    name: str = doc.get("name") or (
        full_name.split("/")[1] if "/" in full_name else full_name
    )

    # language: stored as {name: "Python"} in Kaggle data; _augment_fields
    # copies it to `language` as-is (still an object).
    lang_raw = doc.get("primaryLanguage") or doc.get("language")
    if isinstance(lang_raw, dict):
        language = lang_raw.get("name") or "Unknown"
    elif isinstance(lang_raw, str):
        language = lang_raw
    else:
        language = "Unknown"

    stars = doc.get("stargazerCount") or doc.get("stars") or 0
    forks = doc.get("forkCount") or doc.get("forks") or 0

    url = doc.get("url") or f"https://github.com/{full_name}"

    return {
        "id": doc.get("repo_id") or str(doc.get("_id", full_name)),
        "name": name,
        "owner": owner,
        "description": doc.get("description") or "No description available",
        "stars": stars,
        "forks": forks,
        "language": language,
        "url": url,
    }


@router.post("/lookup")
async def lookup_repos(
    request: Request,
    body: Dict[str, Any] = Body(...),
) -> Dict[str, Any]:
    """Return normalised repo details for a list of repo_ids.

    Body: ``{"repo_ids": ["owner/repo", ...]}``

    Queries ``gitquery.repositories`` by ``_id`` (which equals
    ``nameWithOwner`` / ``repo_id``) and falls back to querying by
    the ``repo_id`` field for documents whose ``_id`` was written differently.
    """
    repo_ids: List[str] = body.get("repo_ids", [])
    if not repo_ids:
        return {"repos": []}

    mongo_client = get_mongo_client()
    if not mongo_client:
        raise HTTPException(status_code=503, detail="MongoDB not available")

    try:
        db = mongo_client.get_database("gitquery")
        coll = db.get_collection("repositories")

        # Primary lookup: _id field (set by kaggle_to_mongo to nameWithOwner)
        docs: List[Dict[str, Any]] = list(
            coll.find({"_id": {"$in": repo_ids}}, limit=len(repo_ids))
        )

        # Secondary lookup for any IDs not found by _id
        found_ids = {str(d["_id"]) for d in docs}
        missing = [rid for rid in repo_ids if rid not in found_ids]
        if missing:
            extra = list(
                coll.find({"repo_id": {"$in": missing}}, limit=len(missing))
            )
            docs.extend(extra)

        # Preserve the original order from repo_ids
        id_to_doc: Dict[str, Dict] = {}
        for doc in docs:
            key = doc.get("repo_id") or str(doc.get("_id", ""))
            id_to_doc[key] = doc

        ordered = [id_to_doc[rid] for rid in repo_ids if rid in id_to_doc]

        return {"repos": [_normalize(d) for d in ordered]}

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lookup failed: {exc}") from exc
