import os

# Override settings before any recommender module imports resolve .env
os.environ.setdefault("MONGODB_URL", "mongodb://localhost:27017/test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")
os.environ.setdefault("AB_TEST_ENABLED", "false")
