import os
import time
import sys
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, OptimizersConfigDiff

time.sleep(5)
try:
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)

for name in ["repository_embeddings", "code_embeddings", "user_preference_embeddings"]:
    try:
        if any(c.name == name for c in client.get_collections().collections):
            continue
    except Exception as e:
        print(f"Error checking collection {name}: {e}")
    try:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            optimizers_config=OptimizersConfigDiff(indexing_threshold=10000),
        )
    except Exception as e:
        print(f"Error creating {name}: {e}")

print(" Qdrant initialized")
