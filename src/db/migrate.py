"""Simple migration runner for document databases.

Usage: import and call `run_migrations(mongo_client)` from deploy/init scripts.
This is intentionally small and synchronous.
"""

from typing import List


MIGRATIONS = [
    ("0001_initial", "db/migrations/0001_initial.py"),
]


def run_migrations(db):
    applied = set()
    if "schema_migrations" in db.list_collection_names():
        for r in db.schema_migrations.find({}, {"_id": 0}):
            applied.add(r.get("id"))

    for mid, path in MIGRATIONS:
        if mid in applied:
            continue
        # Execute migration module
        import importlib.util
        import os

        spec = importlib.util.spec_from_file_location(
            mid, os.path.join(os.path.dirname(__file__), path.replace("db/", ""))
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "apply"):
            mod.apply(db)
