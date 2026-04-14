"""Populate cleaned repositories from raw_repositories with checkpointing.

This script reads pending raw records, transforms them into cleaned schema,
upserts into the destination collection, and marks source records as processed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from processing.pipelines.transformation import DataTransformer  # noqa: E402


def _safe_json(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return None
        return numeric
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_iso_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:
        return None


class EDAAccumulator:
    """Collect lightweight EDA summaries while records are cleaned."""

    def __init__(self) -> None:
        self.docs_seen = 0
        self.numeric_fields = {
            "stars": self._new_numeric_state(),
            "forks": self._new_numeric_state(),
            "topics_count": self._new_numeric_state(),
            "description_length": self._new_numeric_state(),
            "readme_length": self._new_numeric_state(),
            "days_since_update": self._new_numeric_state(),
        }
        self.language_counts: Dict[str, int] = {}
        self.license_counts: Dict[str, int] = {}
        self.relevance_counts: Dict[str, int] = {}
        self.missing_counts = {
            "description": 0,
            "license": 0,
            "readme": 0,
            "language": 0,
            "topics": 0,
            "pushed_at": 0,
        }
        self.duplicate_key_counts = {
            "full_name": {},
            "repo_id": {},
            "name": {},
        }
        self.stale_repos = 0

    def _new_numeric_state(self) -> Dict[str, Any]:
        return {"count": 0, "sum": 0.0, "min": None, "max": None}

    def _update_numeric(self, field: str, value: Any) -> None:
        numeric = _to_float(value)
        if numeric is None:
            return
        state = self.numeric_fields[field]
        state["count"] += 1
        state["sum"] += numeric
        state["min"] = numeric if state["min"] is None else min(state["min"], numeric)
        state["max"] = numeric if state["max"] is None else max(state["max"], numeric)

    def _inc_counter(self, counter: Dict[str, int], key: str) -> None:
        counter[key] = counter.get(key, 0) + 1

    def _track_duplicate_key(self, field: str, value: Any) -> None:
        if value is None:
            return
        key = str(value).strip()
        if not key:
            return
        field_counter = self.duplicate_key_counts[field]
        field_counter[key] = field_counter.get(key, 0) + 1

    def add_record(self, cleaned: Dict[str, Any]) -> None:
        self.docs_seen += 1

        self._update_numeric("stars", cleaned.get("stars"))
        self._update_numeric("forks", cleaned.get("forks"))

        topics = cleaned.get("topics")
        topics_count = len(topics) if isinstance(topics, list) else 0
        self._update_numeric("topics_count", topics_count)

        description = cleaned.get("description")
        description_length = len(description) if isinstance(description, str) else 0
        self._update_numeric("description_length", description_length)

        readme = cleaned.get("readme")
        readme_length = len(readme) if isinstance(readme, str) else 0
        self._update_numeric("readme_length", readme_length)

        pushed_at = _parse_iso_datetime(cleaned.get("pushed_at"))
        if pushed_at is not None:
            days_since_update = max(0, int((datetime.now(timezone.utc) - pushed_at).days))
            self._update_numeric("days_since_update", days_since_update)
            if days_since_update > 730:
                self.stale_repos += 1
        else:
            self.missing_counts["pushed_at"] += 1

        language = str(cleaned.get("language") or "unknown").strip().lower()
        self._inc_counter(self.language_counts, language)

        license_value = cleaned.get("license")
        normalized_license = str(license_value).strip() if license_value else "missing"
        self._inc_counter(self.license_counts, normalized_license)

        relevance = _to_int(cleaned.get("relevance_label"))
        if relevance is not None:
            self._inc_counter(self.relevance_counts, str(relevance))

        if not description:
            self.missing_counts["description"] += 1
        if not license_value:
            self.missing_counts["license"] += 1
        if not readme:
            self.missing_counts["readme"] += 1
        if not cleaned.get("language"):
            self.missing_counts["language"] += 1
        if not isinstance(topics, list) or len(topics) == 0:
            self.missing_counts["topics"] += 1

        self._track_duplicate_key("full_name", cleaned.get("full_name"))
        self._track_duplicate_key("repo_id", cleaned.get("repo_id"))
        self._track_duplicate_key("name", cleaned.get("name"))

    def _top_n(self, counter: Dict[str, int], n: int = 15) -> List[Dict[str, Any]]:
        return [
            {"value": key, "count": count}
            for key, count in sorted(counter.items(), key=lambda kv: kv[1], reverse=True)[:n]
        ]

    def _numeric_summary(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for field, state in self.numeric_fields.items():
            count = state["count"]
            out[field] = {
                "count": count,
                "mean": (state["sum"] / count) if count else None,
                "min": state["min"],
                "max": state["max"],
            }
        return out

    def _duplicate_summary(self) -> Dict[str, int]:
        summary: Dict[str, int] = {}
        for field, counter in self.duplicate_key_counts.items():
            summary[field] = int(sum(max(0, c - 1) for c in counter.values()))
        return summary

    def build_report(self, context: Dict[str, Any], run_stats: Dict[str, Any]) -> Dict[str, Any]:
        total = max(1, self.docs_seen)
        missing_rates = {k: round(v / total, 6) for k, v in self.missing_counts.items()}
        stale_pct = round(self.stale_repos / total, 6)

        return {
            "generated_at": _now_iso(),
            "context": context,
            "scope": {
                "docs_processed_in_run": self.docs_seen,
                "run_stats": run_stats,
            },
            "dataset_overview": {
                "docs_seen": self.docs_seen,
                "missing_counts": self.missing_counts,
                "missing_rates": missing_rates,
                "duplicate_counts": self._duplicate_summary(),
            },
            "numeric_summary": self._numeric_summary(),
            "categorical_top": {
                "language": self._top_n(self.language_counts, n=20),
                "license": self._top_n(self.license_counts, n=20),
                "relevance_label": self._top_n(self.relevance_counts, n=10),
            },
            "recency": {
                "stale_repos_gt_730_days": self.stale_repos,
                "stale_ratio": stale_pct,
            },
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_checkpoint(checkpoint_file: Path) -> Dict[str, Any] | None:
    if not checkpoint_file.exists():
        return None
    with checkpoint_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def _save_checkpoint(checkpoint_file: Path, data: Dict[str, Any]) -> None:
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def _checkpoint_matches(checkpoint: Dict[str, Any], context: Dict[str, Any]) -> bool:
    for key, value in context.items():
        if checkpoint.get("context", {}).get(key) != value:
            return False
    return True


def _stable_id(cleaned: Dict[str, Any], raw: Dict[str, Any]) -> str | None:
    for key in ("repo_id", "full_name", "name"):
        if cleaned.get(key):
            return str(cleaned[key])
    if raw.get("_id") is not None:
        return str(raw["_id"])
    return None


def run_clean_population(
    mongodb_url: str,
    mongodb_db: str,
    source_collection: str,
    dest_collection: str,
    batch_size: int,
    max_batches: int,
    mark_processed: bool,
    checkpoint_file: str,
    resume: bool,
    reset_checkpoint: bool,
    run_eda: bool,
    eda_output_file: str,
    eda_collection: str,
) -> None:
    checkpoint_path = Path(checkpoint_file)
    context = {
        "mongodb_db": mongodb_db,
        "source_collection": source_collection,
        "dest_collection": dest_collection,
        "batch_size": batch_size,
        "mark_processed": mark_processed,
    }

    if reset_checkpoint and checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"Checkpoint reset: {checkpoint_path}", flush=True)

    stats = {
        "batches_executed": 0,
        "fetched": 0,
        "cleaned": 0,
        "saved": 0,
        "errors": 0,
    }

    if resume:
        checkpoint = _load_checkpoint(checkpoint_path)
        if checkpoint and _checkpoint_matches(checkpoint, context):
            for key in stats:
                stats[key] = int(checkpoint.get(key, 0) or 0)
            if checkpoint.get("completed") is True:
                print("Checkpoint shows completed=true; nothing to do.", flush=True)
                return
            print(
                "Resuming from checkpoint: "
                f"batches={stats['batches_executed']} fetched={stats['fetched']} "
                f"cleaned={stats['cleaned']} saved={stats['saved']} errors={stats['errors']}",
                flush=True,
            )
        elif checkpoint:
            print("Checkpoint found but context changed; starting from scratch.", flush=True)

    client = MongoClient(mongodb_url)
    db = client[mongodb_db]
    source: Collection = db[source_collection]
    dest: Collection = db[dest_collection]
    eda_dest: Collection = db[eda_collection]
    transformer = DataTransformer()
    eda_accumulator = EDAAccumulator() if run_eda else None

    try:
        while True:
            if max_batches > 0 and stats["batches_executed"] >= max_batches:
                break

            query = {"$or": [{"processing_status": {"$exists": False}}, {"processing_status": "pending"}]}
            raw_batch: List[Dict[str, Any]] = list(source.find(query, limit=batch_size))

            if not raw_batch:
                break

            stats["batches_executed"] += 1
            stats["fetched"] += len(raw_batch)

            operations: List[UpdateOne] = []
            saved_ids: List[str] = []
            processed_ids: List[Any] = []

            now = datetime.now(timezone.utc)
            for raw in raw_batch:
                raw_id = raw.get("_id")
                if raw_id is not None:
                    processed_ids.append(raw_id)
                try:
                    cleaned = transformer.transform(raw)
                    if not cleaned:
                        continue

                    stable_id = _stable_id(cleaned, raw)
                    if not stable_id:
                        continue

                    cleaned["_id"] = stable_id
                    cleaned["cleaned_at"] = now
                    cleaned["source"] = "processing_pipeline"

                    if eda_accumulator is not None:
                        eda_accumulator.add_record(cleaned)

                    operations.append(
                        UpdateOne(
                            {"_id": stable_id},
                            {
                                "$set": cleaned,
                                "$setOnInsert": {"created_at_pipeline": now},
                            },
                            upsert=True,
                        )
                    )
                    saved_ids.append(stable_id)
                    stats["cleaned"] += 1
                except Exception as exc:
                    stats["errors"] += 1
                    print(f"Error cleaning record {raw_id}: {exc}", flush=True)

            if operations:
                dest.bulk_write(operations, ordered=False)
                stats["saved"] += len(saved_ids)

            if mark_processed and processed_ids:
                source.update_many(
                    {"_id": {"$in": processed_ids}},
                    {"$set": {"processing_status": "completed", "processed_at": now}},
                )

            _save_checkpoint(
                checkpoint_path,
                {
                    "context": context,
                    **stats,
                    "updated_at": _now_iso(),
                    "completed": False,
                },
            )

            print(
                "Batch completed: "
                f"batch={stats['batches_executed']} fetched={len(raw_batch)} "
                f"cleaned_total={stats['cleaned']} saved_total={stats['saved']} errors_total={stats['errors']}",
                flush=True,
            )

        pending_count = source.count_documents(
            {"$or": [{"processing_status": {"$exists": False}}, {"processing_status": "pending"}]}
        )
        cleaned_count = dest.count_documents({})

        _save_checkpoint(
            checkpoint_path,
            {
                "context": context,
                **stats,
                "pending_source_records": int(pending_count),
                "cleaned_collection_total": int(cleaned_count),
                "updated_at": _now_iso(),
                "completed": pending_count == 0,
            },
        )

        print(
            "Completed: "
            f"batches={stats['batches_executed']} fetched={stats['fetched']} "
            f"cleaned={stats['cleaned']} saved={stats['saved']} errors={stats['errors']} "
            f"pending_source_records={pending_count} cleaned_collection_total={cleaned_count}",
            flush=True,
        )

        if run_eda:
            run_context = {
                "mongodb_db": mongodb_db,
                "source_collection": source_collection,
                "dest_collection": dest_collection,
                "batch_size": batch_size,
                "max_batches": max_batches,
                "mark_processed": mark_processed,
            }
            run_stats = {
                **stats,
                "pending_source_records": int(pending_count),
                "cleaned_collection_total": int(cleaned_count),
            }
            report = (eda_accumulator or EDAAccumulator()).build_report(
                context=run_context,
                run_stats=run_stats,
            )

            report_path = Path(eda_output_file)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with report_path.open("w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=_safe_json)

            report_id = f"{dest_collection}:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            eda_dest.update_one(
                {"_id": report_id},
                {
                    "$set": {
                        **report,
                        "report_path": str(report_path),
                        "saved_at": _now_iso(),
                    }
                },
                upsert=True,
            )

            print(
                "EDA report generated: "
                f"docs_seen={report['scope']['docs_processed_in_run']} "
                f"file={report_path} collection={eda_collection} report_id={report_id}",
                flush=True,
            )
    finally:
        client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean raw_repositories into repositories with checkpointing.")
    parser.add_argument(
        "--mongodb-url",
        default=os.getenv("MONGODB_URL", "mongodb://localhost:27017/gitquery?authSource=admin"),
        help="MongoDB connection URL",
    )
    parser.add_argument(
        "--mongodb-db",
        default=os.getenv("MONGODB_DB", "gitquery"),
        help="MongoDB database name",
    )
    parser.add_argument("--source-collection", default="raw_repositories", help="Source collection")
    parser.add_argument("--dest-collection", default="repositories", help="Destination collection")
    parser.add_argument("--batch-size", type=int, default=500, help="Records per batch")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Max batches to process (0 means run until no pending records)",
    )
    parser.add_argument(
        "--no-mark-processed",
        action="store_true",
        help="Do not set processing_status=completed on source records",
    )
    parser.add_argument(
        "--checkpoint-file",
        default=".checkpoints/clean_repositories_population.json",
        help="Path to checkpoint JSON file",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="Delete existing checkpoint and start from scratch",
    )
    parser.add_argument(
        "--no-eda",
        action="store_true",
        help="Disable EDA report generation for this run",
    )
    parser.add_argument(
        "--eda-output-file",
        default=".reports/clean_repositories_eda.json",
        help="Path to save EDA JSON report",
    )
    parser.add_argument(
        "--eda-collection",
        default="repositories_eda_reports",
        help="Mongo collection to store EDA reports",
    )

    args = parser.parse_args()

    run_clean_population(
        mongodb_url=args.mongodb_url,
        mongodb_db=args.mongodb_db,
        source_collection=args.source_collection,
        dest_collection=args.dest_collection,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        mark_processed=not args.no_mark_processed,
        checkpoint_file=args.checkpoint_file,
        resume=args.resume,
        reset_checkpoint=args.reset_checkpoint,
        run_eda=not args.no_eda,
        eda_output_file=args.eda_output_file,
        eda_collection=args.eda_collection,
    )


if __name__ == "__main__":
    main()
