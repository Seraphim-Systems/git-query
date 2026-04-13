"""MLOps module for experiment tracking and monitoring.

Keep package imports lightweight so importing a submodule such as
``dataset_versioner`` does not eagerly import optional third-party stacks.
"""

from __future__ import annotations

from typing import Any

__all__ = ["MLflowTracker", "DriftMonitor"]


def __getattr__(name: str) -> Any:
    if name == "DriftMonitor":
        from .drift_monitor import DriftMonitor

        return DriftMonitor
    if name == "MLflowTracker":
        from .mlflow_tracker import MLflowTracker

        return MLflowTracker
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
