"""Initial migration: create core collections and indexes for MongoDB.

This migration is idempotent and will create collections/indexes if missing.
It records applied migrations in `schema_migrations` collection.
"""

from pymongo import ASCENDING


def apply(db):
    # Ensure schema_migrations exists
    coll = db.get_collection("schema_migrations")
    # Create example collections and indexes
    if "users" not in db.list_collection_names():
        db.create_collection("users")
    db.users.create_index([("user_id", ASCENDING)], unique=True)

    if "repository_activity" not in db.list_collection_names():
        db.create_collection("repository_activity")
    db.repository_activity.create_index([("repo_id", ASCENDING)])
    db.repository_activity.create_index([("timestamp", -1)])

    # Record migration
    coll.update_one(
        {"id": "0001_initial"}, {"$setOnInsert": {"applied_at": True}}, upsert=True
    )
