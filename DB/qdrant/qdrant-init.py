#!/usr/bin/env python3
"""
Qdrant Vector Database Initialization Script
Creates collections for vector embeddings used in repository recommendations
"""

import os
import time
import sys
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, OptimizersConfigDiff

# Wait for Qdrant to be ready
print("Waiting for Qdrant to be ready...")
time.sleep(5)

# Connect to Qdrant
qdrant_host = os.getenv("QDRANT_HOST", "localhost")
qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
qdrant_api_key = os.getenv("QDRANT_API_KEY")

print(f"Connecting to Qdrant at {qdrant_host}:{qdrant_port}")

try:
    client = QdrantClient(
        host=qdrant_host,
        port=qdrant_port,
        api_key=qdrant_api_key if qdrant_api_key else None
    )
except Exception as e:
    print(f"Error connecting to Qdrant: {e}")
    sys.exit(1)

# Collection for repository embeddings
# Using 768 dimensions (typical for models like sentence-transformers)
EMBEDDING_DIM = 768

collections_config = [
    {
        "name": "repository_embeddings",
        "description": "Vector embeddings for repository descriptions and content",
        "vector_size": EMBEDDING_DIM,
        "distance": Distance.COSINE
    },
    {
        "name": "code_embeddings",
        "description": "Vector embeddings for code snippets and README files",
        "vector_size": EMBEDDING_DIM,
        "distance": Distance.COSINE
    },
    {
        "name": "user_preference_embeddings",
        "description": "Vector embeddings for user preferences and interests",
        "vector_size": EMBEDDING_DIM,
        "distance": Distance.COSINE
    }
]

for config in collections_config:
    collection_name = config["name"]
    
    # Check if collection exists
    try:
        collections = client.get_collections().collections
        if any(c.name == collection_name for c in collections):
            print(f"✓ Collection '{collection_name}' already exists, skipping...")
            continue
    except Exception as e:
        print(f"Error checking collections: {e}")
    
    # Create collection
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=config["vector_size"],
                distance=config["distance"]
            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=10000  # Optimize for large datasets
            )
        )
        print(f"✓ Created collection: {collection_name}")
        print(f"  - Vector size: {config['vector_size']}")
        print(f"  - Distance metric: {config['distance']}")
        print(f"  - Description: {config['description']}")
    except Exception as e:
        print(f"✗ Error creating collection '{collection_name}': {e}")
        continue

print("\n✓ Qdrant vector database initialization completed")
print("Collections ready for repository recommendation embeddings")
