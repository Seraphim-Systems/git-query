"""CLI entrypoint for the drift monitor Docker container.

Loads reference and current data from paths specified via environment
variables, runs all applicable drift checks, saves the report, and
exits with code 1 if drift is detected so CI/CD pipelines can react.

Environment variables:
    REFERENCE_DATA_PATH   Path to reference (training) data (.parquet or .json)
    CURRENT_DATA_PATH     Path to current (production) data (.parquet or .json)
    EVIDENTLY_REPORT_PATH Directory to write drift reports (default: /app/drift_reports)
    DRIFT_THRESHOLD       Exit 1 only if drift detected (default: true)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _load_data(path: str):
    import pandas as pd

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix == ".json":
        return pd.read_json(p)

    raise ValueError(f"Unsupported file format: {p.suffix}. Use .parquet or .json")


def main() -> None:
    reference_path = os.getenv("REFERENCE_DATA_PATH")
    current_path = os.getenv("CURRENT_DATA_PATH")
    report_dir = os.getenv("EVIDENTLY_REPORT_PATH", "/app/drift_reports")

    if not reference_path or not current_path:
        logger.error("REFERENCE_DATA_PATH and CURRENT_DATA_PATH environment variables are required.")
        sys.exit(2)

    logger.info(f"Loading reference data from: {reference_path}")
    reference_df = _load_data(reference_path)
    logger.info(f"Reference data shape: {reference_df.shape}")

    logger.info(f"Loading current data from: {current_path}")
    current_df = _load_data(current_path)
    logger.info(f"Current data shape: {current_df.shape}")

    from drift_monitor import DriftMonitor

    monitor = DriftMonitor(report_dir=report_dir)

    logger.info("Running drift checks...")
    report = monitor.run_full_drift_check(
        reference_data=reference_df,
        current_data=current_df,
    )

    drift_detected = report.get("overall_drift_detected", False)
    logger.info(f"Drift detected: {drift_detected}")

    summary_path = Path(report_dir) / "drift_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(
            {
                "drift_detected": drift_detected,
                "timestamp": report.get("timestamp"),
                "checks_run": list(report.get("checks", {}).keys()),
            },
            f,
            indent=2,
        )
    logger.info(f"Summary written to: {summary_path}")

    if drift_detected:
        logger.warning("Drift detected — exiting with code 1.")
        sys.exit(1)

    logger.info("No drift detected.")
    sys.exit(0)


if __name__ == "__main__":
    main()
