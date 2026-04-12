"""Trainer classes for the recommendation system.

Imports are intentionally lazy here. EmbeddingTrainer and
RerankerCrossEncoderTrainer use deep relative imports (``...config``,
``...models``) that only resolve when the full ``recommender`` package is
on sys.path.  In the training Docker container PYTHONPATH=/app so the root
is ``training``, not ``recommender.training`` — eager imports of those
trainers would crash at startup even when only the LGBM trainer is needed.

Import the trainer you need directly from its module instead:
    from training.trainers.reranker_lgbm_trainer import RerankerLGBMTrainer
    from training.trainers.embedding_trainer import EmbeddingTrainer  # full recommender path only
"""
