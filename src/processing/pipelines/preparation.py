"""Server-side data preparation pipeline for raw -> cleaned Mongo records."""

import logging
from typing import Any, Dict, List

from processing.pipelines.ingestion import DataIngestion
from processing.pipelines.transformation import DataTransformer

logger = logging.getLogger(__name__)


async def run_preparation_batch(
    ingestion: DataIngestion,
    transformer: DataTransformer,
    limit: int,
    mark_processed: bool = True,
) -> Dict[str, Any]:
    """Fetch, clean, and persist one batch into the cleaned Mongo collection."""
    stats: Dict[str, int] = {
        "fetched": 0,
        "cleaned": 0,
        "saved": 0,
        "errors": 0,
    }

    cleaned_records: List[Dict[str, Any]] = []
    raw_records = await ingestion.fetch_unprocessed(limit=limit)
    stats["fetched"] = len(raw_records)

    if not raw_records:
        return {
            "stats": stats,
            "cleaned_records": cleaned_records,
            "processed_ids": [],
        }

    for record in raw_records:
        try:
            cleaned = transformer.transform(record)
            if cleaned:
                cleaned_records.append(cleaned)
                stats["cleaned"] += 1
        except Exception as exc:
            logger.error("Error cleaning record %s: %s", record.get("_id"), exc)
            stats["errors"] += 1

    if cleaned_records:
        saved_ids = await ingestion.save_cleaned(cleaned_records)
        stats["saved"] = len(saved_ids)

    processed_ids = [r["_id"] for r in raw_records if "_id" in r]
    if mark_processed and processed_ids:
        await ingestion.mark_as_processed(processed_ids)

    return {
        "stats": stats,
        "cleaned_records": cleaned_records,
        "processed_ids": processed_ids,
    }
