from typing import Dict, Any, List, Optional

import os
import httpx
import uuid

from src.db.adapters.interfaces import VectorRepository
from src.db.clients import get_qdrant_client
import logging


DEFAULT_QDRANT_HOST = os.environ.get("QDRANT_HOST", "qdrant")
DEFAULT_QDRANT_PORT = int(os.environ.get("QDRANT_HTTP_PORT", "6333"))


def _http_url(path: str) -> str:
    return f"http://{DEFAULT_QDRANT_HOST}:{DEFAULT_QDRANT_PORT}{path}"


class QdrantAdapter(VectorRepository):
    """Lightweight Qdrant adapter with client-first and HTTP fallbacks.

    Designed to run inside the gateway container where the Qdrant service
    is reachable at host `qdrant` by default.
    """

    def __init__(self, client=None):
        self.client = client or get_qdrant_client()

    def list_collections(self) -> List[str]:
        if not self.client:
            # HTTP fallback
            r = httpx.get(_http_url("/collections"), timeout=5.0)
            r.raise_for_status()
            data = r.json()
            cols = data.get("collections") or data.get("result") or []
            return [c.get("name") if isinstance(c, dict) else str(c) for c in cols]

        cols = self.client.get_collections()
        out = []
        for c in getattr(cols, "collections", []):
            out.append(
                getattr(c, "name", None)
                or getattr(c, "collection_name", None)
                or str(c)
            )
        return out

    def upsert_points(
        self, collection: str, points: List[Dict[str, Any]], wait: bool = True
    ) -> Dict[str, Any]:
        """Upsert points: prefer qdrant-client, fall back to HTTP (and legacy payloads).

        Points format: [{"id": ..., "vector": [...], "payload": {...}}, ...]
        """
        # Normalize inputs
        pts = []
        for p in points:
            raw_id = p.get("id")
            payload = p.get("payload", {}) or {}

            # Qdrant requires point IDs to be unsigned ints or UUIDs.
            # If caller supplied a non-numeric string ID, convert it to a
            # deterministic UUID so inserts/searches are stable across runs.
            # Preserve original ID in payload under `_orig_id` for lookup.
            final_id: Any = raw_id
            if isinstance(raw_id, str):
                # Accept decimal integer strings as ints
                if raw_id.isdigit():
                    final_id = int(raw_id)
                else:
                    # Try parsing as UUID first
                    try:
                        _ = uuid.UUID(raw_id)
                        final_id = raw_id
                    except Exception:
                        # Generate deterministic UUIDv5 from collection+raw_id
                        gen = uuid.uuid5(uuid.NAMESPACE_URL, f"{collection}:{raw_id}")
                        final_id = str(gen)
                        # Preserve original ID so callers can correlate
                        if "_orig_id" not in payload:
                            payload["_orig_id"] = raw_id

            pts.append({"id": final_id, "vector": p.get("vector"), "payload": payload})

        # Try client upsert where available
        if self.client:
            try:
                # Newer clients accept wait kwarg
                try:
                    self.client.upsert(
                        collection_name=collection, points=pts, wait=wait
                    )
                except TypeError:
                    self.client.upsert(collection_name=collection, points=pts)
                return {"collection": collection, "upserted": len(pts)}
            except Exception:
                # Fall through to HTTP fallback
                pass

        # HTTP fallback: POST /collections/{collection}/points
        url = _http_url(f"/collections/{collection}/points?wait={str(wait).lower()}")
        r = httpx.post(url, json={"points": pts}, timeout=30.0)
        # If server complains about missing legacy fields, try legacy payload
        if r.status_code == 400 and r.text and "missing field `ids`" in r.text:
            ids = [p.get("id") for p in pts]
            vectors = [p.get("vector") for p in pts]
            payloads = [p.get("payload") for p in pts]
            r = httpx.post(
                url,
                json={"ids": ids, "vectors": vectors, "payloads": payloads},
                timeout=30.0,
            )

        # Provide richer error context for debugging: include response body
        if r.status_code >= 400:
            logger = logging.getLogger(__name__)
            try:
                body = r.text
            except Exception:
                body = "<could not read response body>"
            logger.error("Qdrant upsert HTTP %s for %s: %s", r.status_code, url, body)
            raise Exception(f"Qdrant upsert failed: HTTP {r.status_code}: {body}")

        return {"collection": collection, "upserted": len(pts)}

    def search(
        self,
        collection: str,
        vector: List[float],
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Try client methods first
        if self.client:
            try:
                if hasattr(self.client, "search"):
                    res = self.client.search(
                        collection_name=collection,
                        query_vector=vector,
                        limit=limit,
                        query_filter=filter or {},
                    )
                elif hasattr(self.client, "search_points"):
                    res = self.client.search_points(
                        collection_name=collection,
                        query_vector=vector,
                        limit=limit,
                        query_filter=filter or {},
                    )
                else:
                    raise AttributeError("No supported search method on qdrant client")

                hits = []
                for h in res:
                    hits.append(
                        {
                            "id": getattr(h, "id", None),
                            "score": getattr(h, "score", None),
                            "payload": getattr(h, "payload", None),
                        }
                    )
                return {"collection": collection, "count": len(hits), "results": hits}
            except Exception:
                pass

        # HTTP fallback: try search endpoint(s)
        body = {"vector": vector, "top": limit}
        if filter:
            body["filter"] = filter

        for path in (
            f"/collections/{collection}/points/search",
            f"/collections/{collection}/points/scroll",
        ):
            try:
                r = httpx.post(_http_url(path), json=body, timeout=20.0)
                if r.status_code == 404:
                    continue
                r.raise_for_status()
                data = r.json()
                items = (
                    data.get("result") or data.get("points") or data.get("hits") or []
                )
                hits = []
                for it in items:
                    if isinstance(it, dict):
                        pid = it.get("id") or it.get("point_id")
                        score = it.get("score") or it.get("distance")
                        payload = it.get("payload")
                    else:
                        pid = getattr(it, "id", None)
                        score = getattr(it, "score", None)
                        payload = getattr(it, "payload", None)
                    hits.append({"id": pid, "score": score, "payload": payload})
                return {"collection": collection, "count": len(hits), "results": hits}
            except httpx.HTTPStatusError:
                continue

        # If nothing worked, return empty result
        return {"collection": collection, "count": 0, "results": []}

    def delete_collection(self, collection: str) -> Dict[str, Any]:
        if self.client:
            try:
                self.client.delete_collection(collection_name=collection)
                return {"collection": collection, "deleted": True}
            except Exception:
                pass

        # HTTP fallback
        r = httpx.delete(_http_url(f"/collections/{collection}"), timeout=10.0)
        if r.status_code in (200, 204):
            return {"collection": collection, "deleted": True}
        return {"collection": collection, "deleted": False, "error": r.text}

    def delete_points(
        self,
        collection: str,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if ids and self.client:
            try:
                if hasattr(self.client, "delete_points"):
                    self.client.delete_points(collection_name=collection, ids=ids)
                    return {"collection": collection, "deleted": len(ids)}
                try:
                    self.client.delete(collection_name=collection, points=ids)
                    return {"collection": collection, "deleted": len(ids)}
                except Exception:
                    pass
            except Exception:
                pass

        # HTTP fallback for ids or filter
        if ids:
            r = httpx.post(
                _http_url(f"/collections/{collection}/points/delete"),
                json={"ids": ids},
                timeout=10.0,
            )
            if r.status_code == 200:
                return {"collection": collection, "deleted": len(ids)}
            return {"collection": collection, "deleted": 0, "error": r.text}

        if filter:
            r = httpx.post(
                _http_url(f"/collections/{collection}/points/delete"),
                json={"filter": filter},
                timeout=10.0,
            )
            if r.status_code == 200:
                return {"collection": collection, "deleted": "filter_applied"}
            return {"collection": collection, "deleted": 0, "error": r.text}

        return {"collection": collection, "deleted": 0}
