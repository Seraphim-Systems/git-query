"""Abstract base pipeline for all training workflows."""

import abc
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class BasePipeline(abc.ABC):
    """Abstract base class for all training pipelines.

    Defines a four-phase lifecycle: fetch → train → evaluate → register.
    The concrete run() method orchestrates all phases, isolating failures
    so a non-critical phase (e.g. evaluate) does not abort the run.
    """

    @abc.abstractmethod
    async def fetch(self) -> Dict[str, Any]:
        """Fetch training data. Returns a training_data dict."""

    @abc.abstractmethod
    async def train(self, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train the model. Returns a metrics dict."""

    @abc.abstractmethod
    async def evaluate(self, training_data: Dict[str, Any], metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the trained model. Returns updated metrics."""

    @abc.abstractmethod
    async def register(self, metrics: Dict[str, Any]) -> None:
        """Register / persist the trained model."""

    async def run(self) -> Optional[Dict[str, Any]]:
        """Orchestrate all phases. Failures in evaluate/register are logged but non-fatal."""
        logger.info("[%s] Starting pipeline run", self.__class__.__name__)

        training_data = await self.fetch()
        metrics = await self.train(training_data)

        try:
            metrics = await self.evaluate(training_data, metrics)
        except Exception:
            logger.exception("[%s] evaluate() failed — continuing", self.__class__.__name__)

        try:
            await self.register(metrics)
        except Exception:
            logger.exception("[%s] register() failed — continuing", self.__class__.__name__)

        logger.info("[%s] Pipeline run complete", self.__class__.__name__)
        return metrics
