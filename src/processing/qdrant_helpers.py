"""Small utilities for serializing Qdrant objects for API responses.

These helpers centralize defensive checks so API routers can return
stable JSON even when the client library returns objects with
nonstandard attributes across versions.
"""

from typing import Any, Dict


def serialize_collection_description(col: Any) -> Dict[str, Any]:
    """Return a dict with collection summary fields extracted safely.

    Accepts either mapping-like objects or client model instances and
    uses getattr/keys to extract common fields with safe defaults.
    """
    # Prefer mapping protocol if available
    try:
        if hasattr(col, "items"):
            return {
                "name": col.get("name") or col.get("collection_name") or "",
                "vectors_count": col.get("vectors_count") or col.get("vectors") or 0,
                "points_count": col.get("points_count") or col.get("points") or 0,
            }
    except Exception:
        pass

    # Fallback to attribute access
    name = getattr(col, "name", None) or getattr(col, "collection_name", "")
    vectors_count = (
        getattr(col, "vectors_count", None) or getattr(col, "vectors", None) or 0
    )
    points_count = (
        getattr(col, "points_count", None) or getattr(col, "points", None) or 0
    )

    # Some client models nest counts under summary/dictionary fields
    if isinstance(vectors_count, dict):
        vectors_count = vectors_count.get("count", 0)
    if isinstance(points_count, dict):
        points_count = points_count.get("count", 0)

    return {"name": name, "vectors_count": vectors_count, "points_count": points_count}
