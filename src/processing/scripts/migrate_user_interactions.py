"""Backfill legacy gateway user interaction history into canonical Mongo collections.

Migrates from:
- users.interaction_history (legacy embedded array)

Into:
- user_interactions (canonical recommender event log)
- user_preferences.total_interactions (counter warm-up)

This script is idempotent by default when --upsert is enabled (default).
"""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import Any, Dict, Iterable, List

from pymongo import UpdateOne
from pymongo.errors import BulkWriteError

from src.db.config import get_mongodb_db


ACTION_MAP = {
    "click": "click",
    "view": "view",
    "dismiss": "dismiss",
    "thumbs_up": "thumbs_up",
    "thumbs_down": "thumbs_down",
    "save": "save",
    "star": "save",
    "bookmark": "save",
    "clone": "click",
    "fork": "click",
    "open": "click",
}


def _map_interaction_type(action: str) -> str:
    return ACTION_MAP.get((action or "").lower(), "view")


def _batch(items: Iterable[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    chunk: List[Dict[str, Any]] = []
    for item in items:
        chunk.append(item)
        if len(chunk) >= batch_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def _build_event(user_id: str, interaction: Dict[str, Any]) -> Dict[str, Any]:
    repo_id = str(interaction.get("repo_id") or "").strip()
    action = str(interaction.get("action") or "view").strip().lower()
    timestamp = interaction.get("timestamp") or datetime.utcnow()
    if isinstance(timestamp, str):
        try:
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            timestamp = datetime.utcnow()

    metadata = interaction.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {"raw_metadata": metadata}

    return {
        "user_id": user_id,
        "query": str(metadata.get("query") or ""),
        "repo_id": repo_id,
        "interaction_type": _map_interaction_type(action),
        "position_in_results": metadata.get("position_in_results"),
        "variant": str(metadata.get("variant") or "legacy"),
        "timestamp": timestamp,
        "metadata": metadata,
    }


def migrate(
    users_collection: str,
    interactions_collection: str,
    prefs_collection: str,
    batch_size: int,
    limit_users: int | None,
    dry_run: bool,
    upsert: bool,
) -> None:
    db = get_mongodb_db()
    users = db[users_collection]
    interactions = db[interactions_collection]
    prefs = db[prefs_collection]

    interactions.create_index([("user_id", 1), ("timestamp", -1)])
    interactions.create_index([("repo_id", 1)])
    interactions.create_index([("variant", 1), ("timestamp", -1)])
    prefs.create_index([("user_id", 1)], unique=True)

    query = {"interaction_history.0": {"$exists": True}}
    cursor = users.find(query, {"user_id": 1, "interaction_history": 1})
    if limit_users:
        cursor = cursor.limit(limit_users)

    migrated_events = 0
    skipped_events = 0
    processed_users = 0

    for user_doc in cursor:
        user_id = str(user_doc.get("user_id") or "").strip()
        if not user_id:
            skipped_events += len(user_doc.get("interaction_history") or [])
            continue

        legacy_events = user_doc.get("interaction_history") or []
        if not legacy_events:
            continue

        canonical_events = []
        for raw in legacy_events:
            if not isinstance(raw, dict):
                skipped_events += 1
                continue
            event = _build_event(user_id, raw)
            if not event["repo_id"]:
                skipped_events += 1
                continue
            canonical_events.append(event)

        if not canonical_events:
            processed_users += 1
            continue

        if dry_run:
            migrated_events += len(canonical_events)
            processed_users += 1
            continue

        for chunk in _batch(canonical_events, batch_size):
            if upsert:
                ops = [
                    UpdateOne(
                        {
                            "user_id": item["user_id"],
                            "repo_id": item["repo_id"],
                            "timestamp": item["timestamp"],
                            "interaction_type": item["interaction_type"],
                        },
                        {"$set": item},
                        upsert=True,
                    )
                    for item in chunk
                ]
                try:
                    result = interactions.bulk_write(ops, ordered=False)
                    migrated_events += result.upserted_count + result.modified_count
                except BulkWriteError as exc:
                    details = exc.details or {}
                    migrated_events += int(details.get("nUpserted", 0)) + int(details.get("nModified", 0))
            else:
                interactions.insert_many(chunk, ordered=False)
                migrated_events += len(chunk)

        prefs.update_one(
            {"user_id": user_id},
            {
                "$set": {"last_updated": datetime.utcnow()},
                "$setOnInsert": {
                    "user_id": user_id,
                    "language_preferences": {},
                    "topic_preferences": {},
                    "total_interactions": 0,
                },
            },
            upsert=True,
        )

        processed_users += 1

    print(
        f"Migration complete: users={processed_users}, migrated_events={migrated_events}, "
        f"skipped_events={skipped_events}, dry_run={dry_run}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill legacy users.interaction_history to canonical user_interactions."
    )
    parser.add_argument("--users-collection", default="users")
    parser.add_argument("--interactions-collection", default="user_interactions")
    parser.add_argument("--prefs-collection", default="user_preferences")
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--limit-users", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--no-upsert",
        action="store_true",
        help="Insert without deduping by natural key; can create duplicates.",
    )

    args = parser.parse_args()
    migrate(
        users_collection=args.users_collection,
        interactions_collection=args.interactions_collection,
        prefs_collection=args.prefs_collection,
        batch_size=args.batch_size,
        limit_users=args.limit_users,
        dry_run=args.dry_run,
        upsert=not args.no_upsert,
    )


if __name__ == "__main__":
    main()
