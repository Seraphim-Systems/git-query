"""Dataset versioning for reproducible ML experiments.

Computes a short, stable hash of a DataFrame's content so that every
MLflow run records exactly which snapshot of the data it was trained on.

Usage::

    from src.recommender.mlops.dataset_versioner import DatasetVersioner

    versioner = DatasetVersioner()
    version = versioner.compute_version(df)  # e.g. "a3f2c1d8"
    versioner.save_version_metadata(version, df, Path("artifacts/dataset_version.json"))
    meta = versioner.load_version_metadata(Path("artifacts/dataset_version.json"))
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class DatasetVersioner:
    """Computes and persists a content-based version identifier for a DataFrame.

    The version is a short SHA-256 hex digest derived from:
    - Sorted column names
    - DataFrame shape (rows × cols)
    - A sample of cell values (first and last 5 rows, all columns)

    This gives a stable fingerprint that changes whenever the data changes
    but is fast to compute (no full-table hashing).
    """

    def __init__(self, sample_rows: int = 5, digest_length: int = 8) -> None:
        """
        Parameters
        ----------
        sample_rows:
            Number of rows sampled from the head *and* tail of the DataFrame.
        digest_length:
            Number of hex characters to keep from the SHA-256 digest.
        """
        self.sample_rows = sample_rows
        self.digest_length = digest_length

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_version(self, df: Any) -> str:
        """Return a short hex string identifying this DataFrame's content.

        Parameters
        ----------
        df:
            A pandas DataFrame.

        Returns
        -------
        str
            Short hex digest, e.g. ``"a3f2c1d8"``.
        """
        fingerprint = self._build_fingerprint(df)
        digest = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
        return digest[: self.digest_length]

    def save_version_metadata(
        self,
        version: str,
        df: Any,
        path: Path,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Persist version metadata to a JSON file.

        Parameters
        ----------
        version:
            The version string returned by :meth:`compute_version`.
        df:
            The DataFrame that was versioned (used to record shape/columns).
        path:
            Destination file path (created including parents).
        extra:
            Optional additional key-value pairs merged into the metadata.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        meta: dict[str, Any] = {
            "version": version,
            "timestamp": datetime.now(UTC).isoformat(),
            "row_count": len(df),
            "col_count": len(df.columns),
            "feature_cols": sorted(df.columns.tolist()),
        }
        if extra:
            meta.update(extra)

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, default=str)

    def load_version_metadata(self, path: Path) -> dict[str, Any]:
        """Load previously saved version metadata from a JSON file.

        Parameters
        ----------
        path:
            Path to the JSON file written by :meth:`save_version_metadata`.

        Returns
        -------
        dict
            The metadata dictionary.
        """
        with open(Path(path), encoding="utf-8") as fh:
            return json.load(fh)

    def versions_match(self, v1: str, v2: str) -> bool:
        """Return True if two version strings are identical.

        Parameters
        ----------
        v1, v2:
            Version strings returned by :meth:`compute_version`.
        """
        return v1 == v2

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_fingerprint(self, df: Any) -> str:
        """Build a deterministic string representation of the DataFrame."""
        cols = sorted(df.columns.tolist())
        shape = f"{len(df)}x{len(df.columns)}"

        head = df.head(self.sample_rows).to_json(orient="records", default_handler=str)
        tail = df.tail(self.sample_rows).to_json(orient="records", default_handler=str)

        return f"cols={cols}|shape={shape}|head={head}|tail={tail}"
