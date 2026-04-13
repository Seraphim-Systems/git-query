"""Data and model drift monitoring using Evidently AI.

This module provides drift detection for:
1. Data Drift: Changes in repository metadata distributions
2. Embedding Drift: Shifts in embedding space
3. Prediction Drift: Changes in recommendation score distributions
4. Target Drift: Changes in user engagement (CTR, ratings)
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Check if Evidently is available
try:
    import pandas as pd
    from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
    from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric
    from evidently.pipeline.column_mapping import ColumnMapping
    from evidently.report import Report

    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logger.warning("Evidently not installed. Drift monitoring will be disabled.")


class DriftMonitor:
    """
    Monitor for data and model drift in the recommendation system.

    Compares current data against reference (training) data to detect:
    - Distribution shifts in repository features
    - Embedding space drift
    - Changes in prediction patterns
    - User engagement trend changes

    Usage:
        monitor = DriftMonitor(report_dir="./drift_reports")

        # Compare current vs reference data
        report = monitor.check_data_drift(
            reference_data=training_df,
            current_data=production_df
        )

        # Save report
        monitor.save_report(report, "data_drift_report")
    """

    def __init__(
        self,
        report_dir: str = "/app/drift_reports",
        reference_data_path: str | None = None,
    ):
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self.reference_data_path = reference_data_path

        if not EVIDENTLY_AVAILABLE:
            logger.warning("Evidently not available. Drift checks will return empty results.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_to_workspace(self, report: "Report", project_name: str) -> None:
        """Save an Evidently report to the workspace so it appears in the UI."""
        workspace_path = os.getenv("EVIDENTLY_WORKSPACE_PATH")
        if not workspace_path:
            return
        try:
            import uuid

            # Evidently 0.4.30 generates UUID7 for project IDs but its own
            # validation rejects them. Patch uuid7 to emit UUID4 instead.
            try:
                import uuid6

                uuid6.uuid7 = lambda: uuid.uuid4()
            except ImportError:
                pass

            from evidently.ui.workspace import Workspace

            ws = Workspace.create(workspace_path)
            projects = ws.search_project(project_name)
            if projects:
                project = projects[0]
            else:
                project = ws.create_project(project_name)
                self._add_default_panels(project, project_name)
                project.save()

            ws.add_report(project.id, report)
            logger.info("Drift report saved to Evidently workspace project: %s", project_name)
        except Exception as e:
            logger.warning("Failed to save to Evidently workspace: %s", e)

    def _add_default_panels(self, project: Any, project_name: str) -> None:
        """Add default dashboard panels to a newly created Evidently project."""
        try:
            from evidently.renderers.html_widgets import WidgetSize
            from evidently.ui.dashboards import (
                CounterAgg,
                DashboardPanelCounter,
                DashboardPanelPlot,
                PanelValue,
                PlotType,
                ReportFilter,
            )

            is_prediction = "prediction" in project_name

            if is_prediction:
                # Prediction drift: ColumnDriftMetric on recommendation scores
                project.dashboard.add_panel(
                    DashboardPanelCounter(
                        title="Prediction Drift Score (last run)",
                        filter=ReportFilter(metadata_values={}, tag_values=[]),
                        agg=CounterAgg.LAST,
                        value=PanelValue(
                            metric_id="ColumnDriftMetric",
                            field_path="drift_score",
                            legend="Drift score",
                        ),
                        size=WidgetSize.HALF,
                    )
                )
                project.dashboard.add_panel(
                    DashboardPanelPlot(
                        title="Prediction Drift Score Over Time",
                        filter=ReportFilter(metadata_values={}, tag_values=[]),
                        values=[
                            PanelValue(
                                metric_id="ColumnDriftMetric",
                                field_path="drift_score",
                                legend="Drift score",
                            )
                        ],
                        plot_type=PlotType.LINE,
                        size=WidgetSize.FULL,
                    )
                )
            else:
                # Data drift: dataset-level drift flag + per-column drift share
                project.dashboard.add_panel(
                    DashboardPanelCounter(
                        title="Dataset Drift Detected",
                        filter=ReportFilter(metadata_values={}, tag_values=[]),
                        agg=CounterAgg.LAST,
                        value=PanelValue(
                            metric_id="DatasetDriftMetric",
                            field_path="share_of_drifted_columns",
                            legend="Share of drifted columns",
                        ),
                        size=WidgetSize.HALF,
                    )
                )
                project.dashboard.add_panel(
                    DashboardPanelPlot(
                        title="Share of Drifted Columns Over Time",
                        filter=ReportFilter(metadata_values={}, tag_values=[]),
                        values=[
                            PanelValue(
                                metric_id="DatasetDriftMetric",
                                field_path="share_of_drifted_columns",
                                legend="Share drifted",
                            )
                        ],
                        plot_type=PlotType.LINE,
                        size=WidgetSize.FULL,
                    )
                )
        except Exception as e:
            logger.warning("Could not add default dashboard panels: %s", e)

    def _extract_drift_status(self, result: dict[str, Any]) -> bool:
        """Extract drift status from Evidently report.as_dict() output."""
        try:
            for metric in result.get("metrics", []):
                metric_result = metric.get("result", {})
                if metric_result.get("dataset_drift", False):
                    return True
                if metric_result.get("drift_detected", False):
                    return True
            return False
        except Exception:
            return False

    def _extract_drift_status_from_snapshot(self, snapshot: Any) -> bool:
        """Extract drift status from an Evidently snapshot object.

        Handles the snapshot API (metric_results with .value attributes)
        as opposed to the dict-based report.as_dict() output.
        """
        try:
            for metric_result in snapshot.metric_results:
                value = metric_result.value
                if getattr(value, "dataset_drift", False):
                    return True
                if getattr(value, "drift_detected", False):
                    return True
            return False
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Drift checks
    # ------------------------------------------------------------------

    def check_data_drift(
        self,
        reference_data: "pd.DataFrame",
        current_data: "pd.DataFrame",
        column_mapping: Optional["ColumnMapping"] = None,
    ) -> dict[str, Any] | None:
        """Check for data drift between reference and current repo feature distributions."""
        if not EVIDENTLY_AVAILABLE:
            logger.warning("Evidently not available, skipping data drift check")
            return None

        try:
            # Drop training-only columns not present in live data, and columns
            # containing arrays/lists or all-null values.
            TRAINING_ONLY_COLS = {"interaction_score", "query_id", "query_text"}

            def _is_scalar_col(series):
                sample = series.dropna().head(10)
                return not any(isinstance(v, (list, np.ndarray)) for v in sample)

            shared_cols = set(reference_data.columns) & set(current_data.columns)
            scalar_cols = [
                c
                for c in shared_cols
                if c not in TRAINING_ONLY_COLS
                and _is_scalar_col(reference_data[c])
                and reference_data[c].notna().any()
                and pd.api.types.is_numeric_dtype(reference_data[c])
            ]
            reference_data = reference_data[scalar_cols]
            current_data = current_data[scalar_cols]

            # Cap sample size to keep statistical tests fast (KS/PSI on large
            # datasets can take minutes; 2000 rows is more than sufficient for drift detection).
            MAX_ROWS = 2000
            if len(reference_data) > MAX_ROWS:
                reference_data = reference_data.sample(MAX_ROWS, random_state=42)
            if len(current_data) > MAX_ROWS:
                current_data = current_data.sample(MAX_ROWS, random_state=42)

            report = Report(metrics=[DataDriftPreset(stattest="psi", stattest_threshold=0.2)])
            report.run(reference_data=reference_data, current_data=current_data)
            self._save_to_workspace(report, "git-query-data-drift")

            drift_detected = self._extract_drift_status(report.as_dict())

            drift_info = {
                "timestamp": datetime.now().isoformat(),
                "type": "data_drift",
                "reference_rows": len(reference_data),
                "current_rows": len(current_data),
                "drift_detected": drift_detected,
            }

            logger.info("Data drift check complete. Drift detected: %s", drift_info["drift_detected"])
            return drift_info

        except Exception as e:
            logger.error("Error checking data drift: %s", e)
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def check_embedding_drift(
        self,
        reference_embeddings: np.ndarray,
        current_embeddings: np.ndarray,
        sample_size: int = 1000,
    ) -> dict[str, Any] | None:
        """Check for drift in embedding space using DatasetDriftMetric."""
        if not EVIDENTLY_AVAILABLE:
            logger.warning("Evidently not available, skipping embedding drift check")
            return None

        try:
            if len(reference_embeddings) > sample_size:
                idx = np.random.choice(len(reference_embeddings), sample_size, replace=False)
                reference_embeddings = reference_embeddings[idx]

            if len(current_embeddings) > sample_size:
                idx = np.random.choice(len(current_embeddings), sample_size, replace=False)
                current_embeddings = current_embeddings[idx]

            ref_df = pd.DataFrame(
                reference_embeddings,
                columns=[f"emb_{i}" for i in range(reference_embeddings.shape[1])],
            )
            cur_df = pd.DataFrame(
                current_embeddings,
                columns=[f"emb_{i}" for i in range(current_embeddings.shape[1])],
            )

            report = Report(metrics=[DatasetDriftMetric()])
            report.run(reference_data=ref_df, current_data=cur_df)
            self._save_to_workspace(report, "git-query-embedding-drift")

            ref_norms = np.linalg.norm(reference_embeddings, axis=1)
            cur_norms = np.linalg.norm(current_embeddings, axis=1)

            drift_info = {
                "timestamp": datetime.now().isoformat(),
                "type": "embedding_drift",
                "reference_samples": len(reference_embeddings),
                "current_samples": len(current_embeddings),
                "embedding_dim": reference_embeddings.shape[1],
                "drift_detected": self._extract_drift_status(report.as_dict()),
                "norm_stats": {
                    "reference_mean_norm": float(np.mean(ref_norms)),
                    "current_mean_norm": float(np.mean(cur_norms)),
                    "norm_shift": float(np.mean(cur_norms) - np.mean(ref_norms)),
                },
            }

            logger.info("Embedding drift check complete. Drift detected: %s", drift_info["drift_detected"])
            return drift_info

        except Exception as e:
            logger.error("Error checking embedding drift: %s", e)
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def check_prediction_drift(
        self,
        reference_scores: list[float],
        current_scores: list[float],
    ) -> dict[str, Any] | None:
        """Check for drift in LightGBM recommendation scores."""
        if not EVIDENTLY_AVAILABLE:
            logger.warning("Evidently not available, skipping prediction drift check")
            return None

        try:
            MAX_ROWS = 2000
            ref_scores = reference_scores[:MAX_ROWS] if len(reference_scores) > MAX_ROWS else reference_scores
            cur_scores = current_scores[:MAX_ROWS] if len(current_scores) > MAX_ROWS else current_scores
            ref_df = pd.DataFrame({"score": ref_scores})
            cur_df = pd.DataFrame({"score": cur_scores})

            report = Report(metrics=[ColumnDriftMetric(column_name="score", stattest="psi", stattest_threshold=0.2)])
            report.run(reference_data=ref_df, current_data=cur_df)
            self._save_to_workspace(report, "git-query-prediction-drift")

            drift_info = {
                "timestamp": datetime.now().isoformat(),
                "type": "prediction_drift",
                "reference_count": len(reference_scores),
                "current_count": len(current_scores),
                "reference_mean": float(np.mean(reference_scores)),
                "current_mean": float(np.mean(current_scores)),
                "drift_detected": self._extract_drift_status(report.as_dict()),
            }

            logger.info("Prediction drift check complete. Drift detected: %s", drift_info["drift_detected"])
            return drift_info

        except Exception as e:
            logger.error("Error checking prediction drift: %s", e)
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    def check_target_drift(
        self,
        reference_interactions: "pd.DataFrame",
        current_interactions: "pd.DataFrame",
        target_column: str = "clicked",
    ) -> dict[str, Any] | None:
        """Check for CTR drift using TargetDriftPreset on binary interaction labels."""
        if not EVIDENTLY_AVAILABLE:
            logger.warning("Evidently not available, skipping target drift check")
            return None

        try:
            column_mapping = ColumnMapping(target=target_column)

            report = Report(metrics=[TargetDriftPreset()])
            report.run(
                reference_data=reference_interactions,
                current_data=current_interactions,
                column_mapping=column_mapping,
            )
            self._save_to_workspace(report, "git-query-ctr-drift")

            ref_ctr = reference_interactions[target_column].mean() if target_column in reference_interactions else 0
            cur_ctr = current_interactions[target_column].mean() if target_column in current_interactions else 0

            drift_info = {
                "timestamp": datetime.now().isoformat(),
                "type": "target_drift",
                "reference_ctr": float(ref_ctr),
                "current_ctr": float(cur_ctr),
                "ctr_change": float(cur_ctr - ref_ctr),
                "drift_detected": self._extract_drift_status(report.as_dict()),
            }

            logger.info("Target drift check complete. CTR change: %.4f", drift_info["ctr_change"])
            return drift_info

        except Exception as e:
            logger.error("Error checking target drift: %s", e)
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    # ------------------------------------------------------------------
    # Report persistence
    # ------------------------------------------------------------------

    def save_report(
        self,
        drift_info: dict[str, Any],
        report_name: str,
        format: str = "json",
    ) -> str:
        """Save drift report dict to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = self.report_dir / f"{report_name}_{timestamp}.json"

        with open(filepath, "w") as f:
            json.dump(drift_info, f, indent=2, default=str)

        logger.info("Drift report saved to: %s", filepath)
        return str(filepath)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run_full_drift_check(
        self,
        reference_data: Optional["pd.DataFrame"] = None,
        current_data: Optional["pd.DataFrame"] = None,
        reference_embeddings: np.ndarray | None = None,
        current_embeddings: np.ndarray | None = None,
        reference_scores: list[float] | None = None,
        current_scores: list[float] | None = None,
        reference_interactions: Optional["pd.DataFrame"] = None,
        current_interactions: Optional["pd.DataFrame"] = None,
    ) -> dict[str, Any]:
        """Run all available drift checks and return a combined report.

        Each check is skipped when its required inputs are None.
        Sets overall_drift_detected=True if any individual check detects drift.
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "checks": {},
            "overall_drift_detected": False,
        }

        # Data drift
        if reference_data is not None and current_data is not None:
            data_drift = self.check_data_drift(reference_data, current_data)
            if data_drift:
                report["checks"]["data_drift"] = data_drift
                if data_drift.get("drift_detected"):
                    report["overall_drift_detected"] = True

        # Embedding drift
        if reference_embeddings is not None and current_embeddings is not None:
            emb_drift = self.check_embedding_drift(reference_embeddings, current_embeddings)
            if emb_drift:
                report["checks"]["embedding_drift"] = emb_drift
                if emb_drift.get("drift_detected"):
                    report["overall_drift_detected"] = True

        # Prediction drift
        if reference_scores is not None and current_scores is not None:
            pred_drift = self.check_prediction_drift(reference_scores, current_scores)
            if pred_drift:
                report["checks"]["prediction_drift"] = pred_drift
                if pred_drift.get("drift_detected"):
                    report["overall_drift_detected"] = True

        # CTR / target drift
        if reference_interactions is not None and current_interactions is not None:
            ctr_drift = self.check_target_drift(reference_interactions, current_interactions)
            if ctr_drift:
                report["checks"]["ctr_drift"] = ctr_drift
                if ctr_drift.get("drift_detected"):
                    report["overall_drift_detected"] = True

        self.save_report(report, "full_drift_report")
        return report


def create_monitor_from_env() -> DriftMonitor:
    """Create a DriftMonitor from environment variables."""
    return DriftMonitor(
        report_dir=os.getenv("EVIDENTLY_REPORT_PATH", "/app/drift_reports"),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    monitor = create_monitor_from_env()
    logger.info("Drift monitor initialized. Reports → %s", monitor.report_dir)
