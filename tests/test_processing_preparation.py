"""Tests for server-side data preparation pipeline."""

import asyncio

from processing.pipelines.preparation import run_preparation_batch


class _FakeIngestion:
    def __init__(self, records):
        self._records = records
        self.saved = []
        self.processed_ids = []

    async def fetch_unprocessed(self, limit=100):
        return self._records[:limit]

    async def save_cleaned(self, records):
        self.saved.extend(records)
        return [r["repo_id"] for r in records]

    async def mark_as_processed(self, record_ids):
        self.processed_ids.extend(record_ids)


class _FakeTransformer:
    def transform(self, record):
        if record.get("drop"):
            return None
        return {
            "repo_id": record["repo_id"],
            "full_name": record.get("full_name", ""),
            "description": record.get("description", ""),
            "language": record.get("language", ""),
        }


def test_run_preparation_batch_cleans_and_marks_processed():
    records = [
        {"_id": "1", "repo_id": "r1", "full_name": "o/r1", "description": "d", "language": "Python"},
        {"_id": "2", "repo_id": "r2", "drop": True},
        {"_id": "3", "repo_id": "r3", "full_name": "o/r3", "description": "d", "language": "Go"},
    ]
    ingestion = _FakeIngestion(records)
    transformer = _FakeTransformer()

    result = asyncio.run(
        run_preparation_batch(
            ingestion=ingestion,
            transformer=transformer,
            limit=10,
            mark_processed=True,
        )
    )

    assert result["stats"]["fetched"] == 3
    assert result["stats"]["cleaned"] == 2
    assert result["stats"]["saved"] == 2
    assert result["stats"]["errors"] == 0
    assert len(ingestion.saved) == 2
    assert ingestion.processed_ids == ["1", "2", "3"]
