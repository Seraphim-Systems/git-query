"""Migrate users from email-based user_id to UUID-based user_id.

Updates references in:
- users.user_id
- user_interactions.user_id
- user_preferences.user_id

Default mode only migrates users whose user_id equals email or looks like an email.
Run with --dry-run first.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import Dict, Any
from uuid import uuid4

from src.db.config import get_mongodb_db


def _looks_like_email(value: str) -> bool:
    return "@" in value and "." in value


def migrate(limit: int | None, dry_run: bool, include_all_non_uuid: bool) -> None:
    db = get_mongodb_db()
    users = db.users
    interactions = db.user_interactions
    prefs = db.user_preferences

    projection = {"_id": 1, "user_id": 1, "email": 1, "username": 1}
    cursor = users.find({}, projection)
    if limit:
        cursor = cursor.limit(limit)

    scanned = 0
    candidates = 0
    migrated = 0
    skipped = 0

    for user in cursor:
        scanned += 1
        old_user_id = str(user.get("user_id") or "").strip()
        email = str(user.get("email") or "").strip().lower()

        if not old_user_id:
            skipped += 1
            continue

        should_migrate = False
        if include_all_non_uuid:
            # Conservative heuristic: migrate ids that look email-like.
            should_migrate = _looks_like_email(old_user_id)
        else:
            should_migrate = old_user_id == email or _looks_like_email(old_user_id)

        if not should_migrate:
            continue

        candidates += 1
        new_user_id = str(uuid4())

        if dry_run:
            print(
                f"[DRY-RUN] would migrate user email={email} user_id={old_user_id} -> {new_user_id}",
                flush=True,
            )
            continue

        users.update_one(
            {"_id": user["_id"]},
            {
                "$set": {
                    "user_id": new_user_id,
                    "updated_at": datetime.utcnow(),
                    "legacy_user_id": old_user_id,
                }
            },
        )

        interactions.update_many(
            {"user_id": old_user_id},
            {"$set": {"user_id": new_user_id}},
        )

        prefs.update_many(
            {"user_id": old_user_id},
            {"$set": {"user_id": new_user_id, "last_updated": datetime.utcnow()}},
        )

        migrated += 1

    print(
        f"UUID user migration complete: scanned={scanned}, candidates={candidates}, migrated={migrated}, skipped={skipped}, dry_run={dry_run}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate users from email-based IDs to UUID IDs in MongoDB.",
    )
    parser.add_argument("--limit", type=int, help="Limit number of users to scan.")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing.")
    parser.add_argument(
        "--include-all-non-uuid",
        action="store_true",
        help="Also migrate additional email-like user_id values.",
    )

    args = parser.parse_args()
    migrate(
        limit=args.limit,
        dry_run=args.dry_run,
        include_all_non_uuid=args.include_all_non_uuid,
    )


if __name__ == "__main__":
    main()
