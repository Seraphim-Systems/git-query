"""Kaggle → Cosmos DB raw dataset loader.

Downloads a Kaggle dataset via kagglehub, loads a JSON array file,
and upserts documents into Cosmos DB (Mongo API).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import kagglehub
from pymongo import ReplaceOne
from pymongo.errors import BulkWriteError

try:
    import ijson  # type: ignore
except Exception:
    ijson = None


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.storage.db_config import get_cosmos_db  # noqa: E402


def _find_json_files(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*.json") if p.is_file()])


def _iter_json_array(file_path: Path) -> Iterable[Dict[str, Any]]:
    if ijson is not None:
        with file_path.open("rb") as f:
            for item in ijson.items(f, "item"):
                if not isinstance(item, dict):
                    raise ValueError("Encountered a non-object entry in JSON array.")
                yield item
        return

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected a JSON array at the top level.")

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Entry {idx} is not a JSON object.")
        yield item


def _stable_id(doc: Dict[str, Any], id_field: str | None) -> str:
    if id_field and doc.get(id_field):
        return str(doc[id_field])

    for key in ("nameWithOwner", "full_name", "repo_id"):
        if doc.get(key):
            return str(doc[key])

    if doc.get("owner") and doc.get("name"):
        return f"{doc['owner']}/{doc['name']}"

    payload = json.dumps(doc, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(payload).hexdigest()


def _augment_fields(doc: Dict[str, Any]) -> Dict[str, Any]:
    if "full_name" not in doc and doc.get("nameWithOwner"):
        doc["full_name"] = doc["nameWithOwner"]
    if "repo_id" not in doc and doc.get("nameWithOwner"):
        doc["repo_id"] = doc["nameWithOwner"]
    if "language" not in doc and doc.get("primaryLanguage"):
        doc["language"] = doc["primaryLanguage"]

    if "is_fork" not in doc and "isFork" in doc:
        doc["is_fork"] = doc.get("isFork")
    if "is_archived" not in doc and "isArchived" in doc:
        doc["is_archived"] = doc.get("isArchived")

    if "created_at" not in doc and doc.get("createdAt"):
        doc["created_at"] = doc.get("createdAt")
    if "pushed_at" not in doc and doc.get("pushedAt"):
        doc["pushed_at"] = doc.get("pushedAt")

    return doc


def _batch(iterable: Iterable[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    chunk: List[Dict[str, Any]] = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= batch_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def load_to_cosmos(
    dataset: str,
    file_path: str | None,
    collection_name: str,
    batch_size: int,
    id_field: str | None,
    derive_fields: bool,
    max_records: int | None,
    dry_run: bool,
    list_files: bool,
    progress_every: int,
) -> None:
    print(f"Downloading dataset: {dataset}", flush=True)
    dataset_dir = Path(kagglehub.dataset_download(dataset))
    print(f"Dataset downloaded to: {dataset_dir}", flush=True)

    if list_files:
        all_files = sorted([p for p in dataset_dir.rglob("*") if p.is_file()])
        if not all_files:
            print("No files found in dataset.", flush=True)
            return
        print("Files in dataset:", flush=True)
        for p in all_files:
            print(f"- {p.relative_to(dataset_dir)}", flush=True)
        return

    if file_path:
        candidate = Path(file_path)
        if not candidate.is_absolute():
            candidate = dataset_dir / candidate
        if not candidate.exists():
            raise FileNotFoundError(f"JSON file not found: {candidate}")
        json_file = candidate
    else:
        json_files = _find_json_files(dataset_dir)
        if not json_files:
            raise FileNotFoundError("No .json files found in the Kaggle dataset.")
        if len(json_files) > 1:
            found = "\n".join(str(p.relative_to(dataset_dir)) for p in json_files)
            raise ValueError(
                "Multiple JSON files found. Use --file to choose one:\n" + found
            )
        json_file = json_files[0]

    iterator = _iter_json_array(json_file)

    if dry_run:
        count = 0
        sample: List[Dict[str, Any]] = []
        for record in iterator:
            if max_records and count >= max_records:
                break
            doc = dict(record)
            if derive_fields:
                doc = _augment_fields(doc)
            doc["_id"] = _stable_id(doc, id_field)
            if len(sample) < 5:
                sample.append(doc)
            count += 1
            if progress_every and count % progress_every == 0:
                print(f"Parsed {count} records...", flush=True)

        print(
            f"Dry run: parsed {count} records from {json_file} "
            f"for '{collection_name}'.",
            flush=True,
        )
        if sample:
            print("Sample document keys:", sorted(sample[0].keys()), flush=True)
        return

    cosmos_db = get_cosmos_db()
    collection = cosmos_db[collection_name]

    total_upserted = 0
    total_modified = 0
    batch: List[Dict[str, Any]] = []
    processed = 0

    for record in iterator:
        if max_records and processed >= max_records:
            break
        doc = dict(record)
        if derive_fields:
            doc = _augment_fields(doc)
        doc["_id"] = _stable_id(doc, id_field)
        batch.append(doc)
        processed += 1

        if len(batch) >= batch_size:
            ops = [ReplaceOne({"_id": d["_id"]}, d, upsert=True) for d in batch]
            try:
                result = collection.bulk_write(ops, ordered=False)
                total_upserted += result.upserted_count
                total_modified += result.modified_count
            except BulkWriteError as exc:
                result = exc.details
                total_upserted += result.get("nUpserted", 0)
                total_modified += result.get("nModified", 0)
            batch.clear()

        if progress_every and processed % progress_every == 0:
            print(f"Inserted {processed} records...", flush=True)

    if batch:
        ops = [ReplaceOne({"_id": d["_id"]}, d, upsert=True) for d in batch]
        try:
            result = collection.bulk_write(ops, ordered=False)
            total_upserted += result.upserted_count
            total_modified += result.modified_count
        except BulkWriteError as exc:
            result = exc.details
            total_upserted += result.get("nUpserted", 0)
            total_modified += result.get("nModified", 0)

    print(
        f"Completed: upserted={total_upserted}, modified={total_modified}, "
        f"collection={collection_name}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load Kaggle JSON array dataset into Cosmos DB (Mongo API)."
    )
    parser.add_argument("--dataset", required=True, help="Kaggle dataset slug")
    parser.add_argument("--file", help="JSON filename inside the dataset")
    parser.add_argument(
        "--collection",
        default="repositories",
        help="Target Cosmos collection name",
    )
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument(
        "--id-field",
        default="nameWithOwner",
        help="Field used to derive _id when present",
    )
    parser.add_argument(
        "--no-derive",
        action="store_true",
        help="Do not add derived fields like repo_id/full_name",
    )
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--list-files",
        action="store_true",
        help="List files in the Kaggle dataset and exit",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=5000,
        help="Print progress every N records (0 to disable)",
    )

    args = parser.parse_args()

    load_to_cosmos(
        dataset=args.dataset,
        file_path=args.file,
        collection_name=args.collection,
        batch_size=args.batch_size,
        id_field=args.id_field,
        derive_fields=not args.no_derive,
        max_records=args.max_records,
        dry_run=args.dry_run,
        list_files=args.list_files,
        progress_every=args.progress_every,
    )


if __name__ == "__main__":
    main()
