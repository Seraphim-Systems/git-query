"""Kaggle → database raw dataset loader.

Downloads a Kaggle dataset via kagglehub, loads a JSON array file,
and upserts documents into MongoDB/Cosmos DB.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import kagglehub
import httpx
from pymongo import ReplaceOne
from pymongo.errors import BulkWriteError

try:
    import ijson  # type: ignore
except Exception:
    ijson = None


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Avoid importing Cosmos DB config unless direct mode is used.


def _now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


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


def _is_duplicate_error(exc: httpx.HTTPStatusError) -> bool:
    status = exc.response.status_code
    body = (exc.response.text or "").lower()
    return status == 409 or "duplicate key" in body or "e11000" in body or "already exists" in body


def _post_gateway_bulk_with_retry(
    client: httpx.Client,
    bulk_url: str,
    headers: Dict[str, str],
    database_name: str,
    documents: List[Dict[str, Any]],
) -> Dict[str, int]:
    payload = {
        "documents": documents,
        "ordered": False,
        "upsert": True,
        "database": database_name,
    }

    try:
        response = client.post(bulk_url, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return {
            "upserted": int(data.get("upserted") or 0),
            "modified": int(data.get("modified") or 0),
            "failed_docs": 0,
            "duplicate_docs": 0,
        }
    except httpx.HTTPStatusError as exc:
        if _is_duplicate_error(exc):
            return {
                "upserted": 0,
                "modified": 0,
                "failed_docs": 0,
                "duplicate_docs": len(documents),
            }

        if exc.response.status_code < 500 or len(documents) <= 1:
            bad_id = documents[0].get("_id") if documents else None
            print(
                f"Skipping document due to gateway error status={exc.response.status_code}, _id={bad_id}",
                flush=True,
            )
            return {
                "upserted": 0,
                "modified": 0,
                "failed_docs": len(documents),
                "duplicate_docs": 0,
            }

        mid = len(documents) // 2
        left = _post_gateway_bulk_with_retry(client, bulk_url, headers, database_name, documents[:mid])
        right = _post_gateway_bulk_with_retry(client, bulk_url, headers, database_name, documents[mid:])
        return {
            "upserted": left["upserted"] + right["upserted"],
            "modified": left["modified"] + right["modified"],
            "failed_docs": left["failed_docs"] + right["failed_docs"],
            "duplicate_docs": left["duplicate_docs"] + right["duplicate_docs"],
        }
    except httpx.TransportError as exc:
        if len(documents) <= 1:
            bad_id = documents[0].get("_id") if documents else None
            print(
                f"Skipping document due to connection error ({exc}), _id={bad_id}",
                flush=True,
            )
            return {
                "upserted": 0,
                "modified": 0,
                "failed_docs": len(documents),
                "duplicate_docs": 0,
            }
        print(
            f"Connection error ({exc}) on batch of {len(documents)}, splitting and retrying...",
            flush=True,
        )
        mid = len(documents) // 2
        left = _post_gateway_bulk_with_retry(client, bulk_url, headers, database_name, documents[:mid])
        right = _post_gateway_bulk_with_retry(client, bulk_url, headers, database_name, documents[mid:])
        return {
            "upserted": left["upserted"] + right["upserted"],
            "modified": left["modified"] + right["modified"],
            "failed_docs": left["failed_docs"] + right["failed_docs"],
            "duplicate_docs": left["duplicate_docs"] + right["duplicate_docs"],
        }


def _post_gateway_insert_with_retry(
    client: httpx.Client,
    insert_url: str,
    headers: Dict[str, str],
    database_name: str,
    collection_name: str,
    documents: List[Dict[str, Any]],
) -> Dict[str, int]:
    payload = {
        "database": database_name,
        "collection": collection_name,
        "documents": documents,
    }
    try:
        response = client.post(insert_url, json=payload, headers=headers)
        response.raise_for_status()
        return {"upserted": 0, "modified": 0, "failed_docs": 0, "duplicate_docs": 0}
    except httpx.HTTPStatusError as exc:
        if _is_duplicate_error(exc):
            return {
                "upserted": 0,
                "modified": 0,
                "failed_docs": 0,
                "duplicate_docs": len(documents),
            }

        if exc.response.status_code < 500 or len(documents) <= 1:
            bad_id = documents[0].get("_id") if documents else None
            print(
                f"Skipping document due to gateway insert error status={exc.response.status_code}, _id={bad_id}",
                flush=True,
            )
            return {
                "upserted": 0,
                "modified": 0,
                "failed_docs": len(documents),
                "duplicate_docs": 0,
            }
        mid = len(documents) // 2
        left = _post_gateway_insert_with_retry(
            client, insert_url, headers, database_name, collection_name, documents[:mid]
        )
        right = _post_gateway_insert_with_retry(
            client, insert_url, headers, database_name, collection_name, documents[mid:]
        )
        return {
            "upserted": 0,
            "modified": 0,
            "failed_docs": left["failed_docs"] + right["failed_docs"],
            "duplicate_docs": left["duplicate_docs"] + right["duplicate_docs"],
        }
    except httpx.TransportError as exc:
        if len(documents) <= 1:
            bad_id = documents[0].get("_id") if documents else None
            print(
                f"Skipping document due to connection error ({exc}), _id={bad_id}",
                flush=True,
            )
            return {
                "upserted": 0,
                "modified": 0,
                "failed_docs": len(documents),
                "duplicate_docs": 0,
            }
        print(
            f"Connection error ({exc}) on batch of {len(documents)}, splitting and retrying...",
            flush=True,
        )
        mid = len(documents) // 2
        left = _post_gateway_insert_with_retry(
            client, insert_url, headers, database_name, collection_name, documents[:mid]
        )
        right = _post_gateway_insert_with_retry(
            client, insert_url, headers, database_name, collection_name, documents[mid:]
        )
        return {
            "upserted": 0,
            "modified": 0,
            "failed_docs": left["failed_docs"] + right["failed_docs"],
            "duplicate_docs": left["duplicate_docs"] + right["duplicate_docs"],
        }


def load_to_cosmos(
    dataset: str,
    file_path: str | None,
    collection_name: str,
    database_name: str,
    batch_size: int,
    id_field: str | None,
    derive_fields: bool,
    max_records: int | None,
    dry_run: bool,
    list_files: bool,
    progress_every: int,
    gateway_url: str | None,
    api_key: str | None,
    target: str,
    gateway_bulk_path: str | None,
    gateway_mongo_mode: str,
    checkpoint_file: str,
    resume: bool,
    reset_checkpoint: bool,
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
            raise ValueError("Multiple JSON files found. Use --file to choose one:\n" + found)
        json_file = json_files[0]

    iterator = _iter_json_array(json_file)

    checkpoint_path = Path(checkpoint_file)
    context = {
        "dataset": dataset,
        "file": str(json_file),
        "collection": collection_name,
        "database": database_name,
        "target": target,
        "id_field": id_field,
        "derive_fields": derive_fields,
        "batch_size": batch_size,
    }

    if reset_checkpoint and checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"Checkpoint reset: {checkpoint_path}", flush=True)

    resume_from = 0
    total_upserted = 0
    total_modified = 0
    total_failed_docs = 0
    total_duplicate_docs = 0

    if resume and not dry_run:
        checkpoint = _load_checkpoint(checkpoint_path)
        if checkpoint and _checkpoint_matches(checkpoint, context):
            resume_from = int(checkpoint.get("processed", 0) or 0)
            total_upserted = int(checkpoint.get("upserted", 0) or 0)
            total_modified = int(checkpoint.get("modified", 0) or 0)
            total_failed_docs = int(checkpoint.get("failed_docs", 0) or 0)
            total_duplicate_docs = int(checkpoint.get("duplicate_docs", 0) or 0)
            print(
                f"Resuming from checkpoint: processed={resume_from}, upserted={total_upserted}, modified={total_modified}",
                flush=True,
            )
        elif checkpoint:
            print("Checkpoint found but context changed; starting from scratch.", flush=True)

    if resume_from:
        skipped = 0
        while skipped < resume_from:
            try:
                next(iterator)
                skipped += 1
                if progress_every and skipped % progress_every == 0:
                    print(f"Skipped {skipped} records to resume position...", flush=True)
            except StopIteration:
                print("Checkpoint offset is past EOF; nothing to process.", flush=True)
                return

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
            f"Dry run: parsed {count} records from {json_file} for '{collection_name}'.",
            flush=True,
        )
        if sample:
            print("Sample document keys:", sorted(sample[0].keys()), flush=True)
        return

    use_gateway = bool(gateway_url)
    if use_gateway:
        base_url = gateway_url.rstrip("/")
        if target == "mongodb" and gateway_mongo_mode == "insert":
            bulk_url = f"{base_url}/api/mongodb/insert"
        elif gateway_bulk_path:
            bulk_url = f"{base_url}{gateway_bulk_path.format(target=target, collection=collection_name)}"
        elif target == "mongodb":
            bulk_url = f"{base_url}/api/mongodb/collections/{collection_name}/bulk"
        else:
            bulk_url = f"{base_url}/api/v1/db/{target}/{collection_name}/bulk"
        headers = {"X-API-Key": api_key} if api_key else {}
        client = httpx.Client(timeout=300.0)
    else:
        if target == "mongodb":
            from src.db.config import get_mongodb_db  # noqa: E402

            database = get_mongodb_db()
        else:
            raise RuntimeError(
                "Direct Cosmos mode is no longer supported in this repository. "
                "Use --target mongodb (direct or gateway mode)."
            )
        collection = database[collection_name]

    batch: List[Dict[str, Any]] = []
    processed = resume_from

    def persist_checkpoint() -> None:
        if dry_run:
            return
        payload = {
            "context": context,
            "processed": processed,
            "upserted": total_upserted,
            "modified": total_modified,
            "failed_docs": total_failed_docs,
            "duplicate_docs": total_duplicate_docs,
            "updated_at": _now_iso(),
            "completed": False,
        }
        _save_checkpoint(checkpoint_path, payload)

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
            if use_gateway:
                if target == "mongodb" and gateway_mongo_mode == "insert":
                    result = _post_gateway_insert_with_retry(
                        client,
                        bulk_url,
                        headers,
                        database_name,
                        collection_name,
                        batch,
                    )
                else:
                    result = _post_gateway_bulk_with_retry(client, bulk_url, headers, database_name, batch)
                total_upserted += result["upserted"]
                total_modified += result["modified"]
                total_failed_docs += result["failed_docs"]
                total_duplicate_docs += result["duplicate_docs"]
            else:
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
            persist_checkpoint()

        if progress_every and processed % progress_every == 0:
            print(f"Inserted {processed} records...", flush=True)

    if batch:
        if use_gateway:
            if target == "mongodb" and gateway_mongo_mode == "insert":
                result = _post_gateway_insert_with_retry(
                    client,
                    bulk_url,
                    headers,
                    database_name,
                    collection_name,
                    batch,
                )
            else:
                result = _post_gateway_bulk_with_retry(client, bulk_url, headers, database_name, batch)
            total_upserted += result["upserted"]
            total_modified += result["modified"]
            total_failed_docs += result["failed_docs"]
            total_duplicate_docs += result["duplicate_docs"]
        else:
            ops = [ReplaceOne({"_id": d["_id"]}, d, upsert=True) for d in batch]
            try:
                result = collection.bulk_write(ops, ordered=False)
                total_upserted += result.upserted_count
                total_modified += result.modified_count
            except BulkWriteError as exc:
                result = exc.details
                total_upserted += result.get("nUpserted", 0)
                total_modified += result.get("nModified", 0)
        persist_checkpoint()

    if use_gateway:
        client.close()

    if not dry_run:
        _save_checkpoint(
            checkpoint_path,
            {
                "context": context,
                "processed": processed,
                "upserted": total_upserted,
                "modified": total_modified,
                "failed_docs": total_failed_docs,
                "duplicate_docs": total_duplicate_docs,
                "updated_at": _now_iso(),
                "completed": True,
            },
        )

    print(
        f"Completed: upserted={total_upserted}, modified={total_modified}, duplicates={total_duplicate_docs}, failed_docs={total_failed_docs}, "
        f"collection={collection_name}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Load Kaggle JSON array dataset into MongoDB/Cosmos DB.")
    parser.add_argument("--dataset", required=True, help="Kaggle dataset slug")
    parser.add_argument("--file", help="JSON filename inside the dataset")
    parser.add_argument(
        "--collection",
        default="repositories",
        help="Target collection name",
    )
    parser.add_argument(
        "--database",
        default="gitquery",
        help="Target database name",
    )
    parser.add_argument(
        "--target",
        choices=["mongodb", "cosmos"],
        default="mongodb",
        help="Target backend for inserts",
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
    parser.add_argument(
        "--gateway-url",
        help="Gateway base URL for bulk insert (e.g., https://host)",
    )
    parser.add_argument(
        "--api-key",
        help="API key for gateway (sent as X-API-Key)",
    )
    parser.add_argument(
        "--gateway-bulk-path",
        help=("Custom gateway bulk path template, e.g. '/api/mongodb/collections/{collection}/bulk'"),
    )
    parser.add_argument(
        "--gateway-mongo-mode",
        choices=["insert", "bulk"],
        default="insert",
        help="Gateway mode for MongoDB target",
    )
    parser.add_argument(
        "--checkpoint-file",
        default=".checkpoints/raw_repositories_population.json",
        help="Path to checkpoint JSON file",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if present",
    )
    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="Delete existing checkpoint and start from scratch",
    )

    args = parser.parse_args()

    load_to_cosmos(
        dataset=args.dataset,
        file_path=args.file,
        collection_name=args.collection,
        database_name=args.database,
        batch_size=args.batch_size,
        id_field=args.id_field,
        derive_fields=not args.no_derive,
        max_records=args.max_records,
        dry_run=args.dry_run,
        list_files=args.list_files,
        progress_every=args.progress_every,
        gateway_url=args.gateway_url,
        api_key=args.api_key,
        target=args.target,
        gateway_bulk_path=args.gateway_bulk_path,
        gateway_mongo_mode=args.gateway_mongo_mode,
        checkpoint_file=args.checkpoint_file,
        resume=args.resume,
        reset_checkpoint=args.reset_checkpoint,
    )


if __name__ == "__main__":
    main()
