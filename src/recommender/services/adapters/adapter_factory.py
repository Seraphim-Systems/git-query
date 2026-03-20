"""Factory that selects the right adapter based on the model file extension."""

import logging
import pathlib

from .base_adapter import BaseRerankerAdapter

logger = logging.getLogger(__name__)

_LGBM_EXTENSIONS = {".pkl", ".joblib"}


class AdapterFactory:
    """Returns the appropriate BaseRerankerAdapter for a model path or name."""

    @staticmethod
    def from_path(path: str) -> BaseRerankerAdapter:
        """Instantiate the correct adapter.

        LightGBM artifacts end in .pkl or .joblib.
        Everything else is treated as a CrossEncoder model id/path.
        """
        suffix = pathlib.Path(path).suffix.lower()
        if suffix in _LGBM_EXTENSIONS:
            logger.info("Detected LightGBM artifact (%s) — using LGBMAdapter", suffix)
            from .lgbm_adapter import LGBMAdapter

            return LGBMAdapter(path)

        logger.info("Using CrossEncoderAdapter for path: %s", path)
        from .cross_encoder_adapter import CrossEncoderAdapter

        return CrossEncoderAdapter(path)
