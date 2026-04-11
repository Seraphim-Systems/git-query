"""CLI entrypoint for the drift monitor Docker container.

Loads reference data from disk, fetches current data live from MongoDB,
runs all applicable drift checks (data, embedding, CTR), saves reports,
and exits with code 1 if drift is detected so the CI pipeline can react.

Environment variables (required):
    REFERENCE_DATA_PATH       Path to reference parquet saved after last training run

Environment variables (optional — enable additional checks):
    API_BASE_URL              MongoDB gateway URL (enables live current-data fetch + CTR)
    APIKEY_MONGODB            MongoDB/gateway API key
    QDRANT_URL                Qdrant service URL (enables embedding drift)
    APIKEY_QDRANT             Qdrant API key (falls back to APIKEY_MONGODB)
    QDRANT_COLLECTION         Qdrant collection name (default: repositories_embeddings)
    REFERENCE_EMBEDDINGS_PATH Path to reference embeddings .npy file
    CURRENT_DATA_PATH         Path to current data parquet (alternative to live fetch)
    EVIDENTLY_REPORT_PATH     Directory for JSON drift reports (default: /app/drift_reports)
    EVIDENTLY_WORKSPACE_PATH  Evidently workspace path for UI display

Exit codes:
    0 — no drift detected (or insufficient data to evaluate)
    1 — drift detected
    2 — missing required environment variables or data
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------


def _load_parquet(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".json":
        return pd.read_json(p)
    raise ValueError(f"Unsupported format: {p.suffix}. Use .parquet or .json")


def _fetch_current_repos(api_url: str, api_key: str, max_repos: int = 50_000) -> pd.DataFrame:
    """Fetch a sample of current repos from MongoDB as the live current dataset."""
    import requests

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    all_docs: list[dict] = []
    batch_size = 5_000

    for skip in range(0, max_repos, batch_size):
        limit = min(batch_size, max_repos - skip)
        try:
            resp = requests.post(
                f"{api_url.rstrip('/')}/api/mongodb/query",
                headers=headers,
                json={
                    "database": "gitquery",
                    "collection": "repositories",
                    "filter": {},
                    "limit": limit,
                    "skip": skip,
                    "sort": {"_id": 1},
                },
                timeout=60,
            )
            resp.raise_for_status()
            batch = resp.json().get("documents", [])
            if not batch:
                break
            all_docs.extend(batch)
            logger.info("Fetched %d repos so far...", len(all_docs))
        except Exception as e:
            logger.warning("Repo fetch error at skip=%d: %s", skip, e)
            break

    logger.info("Fetched %d total current repos from MongoDB", len(all_docs))
    return pd.DataFrame(all_docs) if all_docs else pd.DataFrame()


def _fetch_current_embeddings(
    qdrant_url: str,
    api_key: str,
    collection: str = "repositories_embeddings",
    sample_size: int = 2_000,
) -> np.ndarray | None:
    """Sample current embedding vectors from Qdrant for embedding drift detection."""
    try:
        from qdrant_client import QdrantClient

        client = QdrantClient(url=qdrant_url, api_key=api_key, timeout=60)
        points, _ = client.scroll(
            collection_name=collection,
            limit=sample_size,
            with_vectors=True,
        )
        if not points:
            logger.warning("No points returned from Qdrant collection '%s'", collection)
            return None
        vectors = np.array([p.vector for p in points], dtype=np.float32)
        logger.info("Fetched %d current embeddings from Qdrant (shape=%s)", len(vectors), vectors.shape)
        return vectors
    except Exception as e:
        logger.warning("Could not fetch current embeddings from Qdrant: %s", e)
        return None


def _fetch_interactions_for_ctr(
    api_url: str,
    api_key: str,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Fetch user interactions and split into (reference, current) DataFrames.

    Derives a binary ``clicked`` column:
      - click / save / thumbs_up  → 1
      - dismiss / thumbs_down / other → 0

    Splits the interaction log in half by time so reference = older half,
    current = newer half.  Returns (None, None) when fewer than 20 interactions
    exist — not enough for a meaningful statistical test.
    """
    import requests

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    try:
        resp = requests.post(
            f"{api_url.rstrip('/')}/api/mongodb/query",
            headers=headers,
            json={
                "database": "gitquery",
                "collection": "user_interactions",
                "filter": {},
                "limit": 100_000,
                "skip": 0,
            },
            timeout=30,
        )
        resp.raise_for_status()
        docs = resp.json().get("documents", [])
    except Exception as e:
        logger.warning("Could not fetch interactions for CTR drift: %s", e)
        return None, None

    if len(docs) < 20:
        logger.info("Too few interactions (%d) for CTR drift — skipping", len(docs))
        return None, None

    df = pd.DataFrame(docs)
    positive_types = {"click", "save", "thumbs_up"}
    df["clicked"] = df.get("interaction_type", pd.Series(dtype=str)).isin(positive_types).astype(int)

    # Sort by timestamp if available so the split is chronological
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.sort_values("timestamp")

    mid = len(df) // 2
    ref_df = df.iloc[:mid][["clicked"]].reset_index(drop=True)
    cur_df = df.iloc[mid:][["clicked"]].reset_index(drop=True)

    logger.info(
        "CTR split: reference=%d rows (CTR=%.3f), current=%d rows (CTR=%.3f)",
        len(ref_df),
        ref_df["clicked"].mean(),
        len(cur_df),
        cur_df["clicked"].mean(),
    )
    return ref_df, cur_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    reference_path = os.getenv("REFERENCE_DATA_PATH")
    current_path = os.getenv("CURRENT_DATA_PATH")
    api_url = os.getenv("API_BASE_URL")
    api_key = os.getenv("APIKEY_MONGODB")
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_key = os.getenv("APIKEY_QDRANT") or api_key
    qdrant_collection = os.getenv("QDRANT_COLLECTION", "repositories_embeddings")
    embeddings_path = os.getenv("REFERENCE_EMBEDDINGS_PATH")
    report_dir = os.getenv("EVIDENTLY_REPORT_PATH", "/app/drift_reports")

    if not reference_path:
        logger.error("REFERENCE_DATA_PATH is required.")
        sys.exit(2)

    # --- Reference data (features from last training run) ---
    logger.info("Loading reference data from: %s", reference_path)
    try:
        reference_df = _load_parquet(reference_path)
    except FileNotFoundError as e:
        logger.error("%s", e)
        sys.exit(2)
    logger.info("Reference data shape: %s", reference_df.shape)

    # --- Current data (live repos from MongoDB or file) ---
    current_df: pd.DataFrame | None = None
    if current_path and Path(current_path).exists():
        logger.info("Loading current data from file: %s", current_path)
        current_df = _load_parquet(current_path)
    elif api_url and api_key:
        logger.info("Fetching current repos from MongoDB...")
        current_df = _fetch_current_repos(api_url, api_key)
    else:
        logger.error("Provide either CURRENT_DATA_PATH or both API_BASE_URL + APIKEY_MONGODB.")
        sys.exit(2)

    if current_df is None or current_df.empty:
        logger.error("Current data is empty — cannot run drift check.")
        sys.exit(2)
    logger.info("Current data shape: %s", current_df.shape)

    # --- Reference embeddings (saved by EmbeddingIndexingPipeline) ---
    reference_embeddings: np.ndarray | None = None
    if embeddings_path and Path(embeddings_path).exists():
        reference_embeddings = np.load(embeddings_path).astype(np.float32)
        logger.info("Loaded reference embeddings: shape=%s", reference_embeddings.shape)
    else:
        logger.info("REFERENCE_EMBEDDINGS_PATH not set or not found — skipping embedding drift")

    # --- Current embeddings (sampled from Qdrant) ---
    current_embeddings: np.ndarray | None = None
    if reference_embeddings is not None and qdrant_url and qdrant_key:
        current_embeddings = _fetch_current_embeddings(qdrant_url, qdrant_key, qdrant_collection)
    elif reference_embeddings is not None:
        logger.info("QDRANT_URL not set — skipping embedding drift")

    # --- Interactions for CTR drift ---
    ref_interactions, cur_interactions = None, None
    if api_url and api_key:
        ref_interactions, cur_interactions = _fetch_interactions_for_ctr(api_url, api_key)

    # --- Run all checks ---
    from drift_monitor import DriftMonitor

    monitor = DriftMonitor(report_dir=report_dir)
    logger.info("Running drift checks...")

    report = monitor.run_full_drift_check(
        reference_data=reference_df,
        current_data=current_df,
        reference_embeddings=reference_embeddings,
        current_embeddings=current_embeddings,
        reference_interactions=ref_interactions,
        current_interactions=cur_interactions,
    )

    checks_run = list(report.get("checks", {}).keys())
    drift_detected = report.get("overall_drift_detected", False)
    logger.info("Checks run: %s", checks_run)
    logger.info("Overall drift detected: %s", drift_detected)

    # --- Write summary ---
    summary_path = Path(report_dir) / "drift_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(
            {
                "drift_detected": drift_detected,
                "timestamp": report.get("timestamp"),
                "checks_run": checks_run,
            },
            f,
            indent=2,
        )
    logger.info("Summary written to: %s", summary_path)

    if drift_detected:
        logger.warning("Drift detected — exiting with code 1.")
        sys.exit(1)

    logger.info("No drift detected.")
    sys.exit(0)


if __name__ == "__main__":
    main()
