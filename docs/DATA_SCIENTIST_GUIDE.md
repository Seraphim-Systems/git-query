# Data Scientist Guide: Database Access and Data Pipelines

This guide explains how data scientists can query databases and set up data ingestion pipelines for the Git-Query project.

## Table of Contents
1. [Quick Start](#quick-start)
2. [API Endpoints](#api-endpoints)
3. [Authentication](#authentication)
4. [Database Access](#database-access)
5. [Data Ingestion Pipelines](#data-ingestion-pipelines)
6. [Python Examples](#python-examples)
7. [Best Practices](#best-practices)

## Quick Start

### Accessing the API

The database API is accessible at: `https://${SERVER_NAME}/api/`

Interactive API documentation is available at:
- Swagger UI: `https://${SERVER_NAME}/docs`
- ReDoc: `https://${SERVER_NAME}/redoc`

### Health Check

```bash
curl https://${SERVER_NAME}/health
```

## API Endpoints

### MongoDB

#### Query Data (No Authentication Required)
```bash
POST /api/mongodb/query
```

Example:
```bash
curl -X POST https://${SERVER_NAME}/api/mongodb/query \
  -H "Content-Type: application/json" \
  -d '{
    "collection": "users",
    "filter": {"username": "johndoe"},
    "limit": 10
  }'
```

#### Insert Data (Authentication Required)
```bash
POST /api/mongodb/insert
```

Example:
```bash
curl -X POST https://${SERVER_NAME}/api/mongodb/insert \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "collection": "users",
    "documents": [
      {"username": "alice", "email": "alice@example.com"},
      {"username": "bob", "email": "bob@example.com"}
    ]
  }'
```

#### List Collections
```bash
GET /api/mongodb/collections?database=gitquery
```

### Redis

#### Get Key
```bash
GET /api/redis/get/{key}
```

#### Set Key (Authentication Required)
```bash
POST /api/redis/set
```

Example:
```bash
curl -X POST https://${SERVER_NAME}/api/redis/set \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "key": "user:123:session",
    "value": "session_token_here",
    "expire": 3600
  }'
```

#### List Keys
```bash
GET /api/redis/keys?pattern=user:*&limit=100
```

### Qdrant (Vector Database)

#### Search Vectors
```bash
POST /api/qdrant/search
```

Example:
```bash
curl -X POST https://${SERVER_NAME}/api/qdrant/search \
  -H "Content-Type: application/json" \
  -d '{
    "collection": "repository_embeddings",
    "vector": [0.1, 0.2, ...],  # 768-dimensional vector
    "limit": 5,
    "score_threshold": 0.7
  }'
```

#### Insert Vectors (Authentication Required)
```bash
POST /api/qdrant/insert
```

Example:
```bash
curl -X POST https://${SERVER_NAME}/api/qdrant/insert \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "collection": "repository_embeddings",
    "points": [
      {
        "id": 1,
        "vector": [0.1, 0.2, ...],
        "payload": {"repo_name": "example/repo", "stars": 1000}
      }
    ]
  }'
```

#### List Collections
```bash
GET /api/qdrant/collections
```

### Batch Operations

#### Batch Insert (Authentication Required)
```bash
POST /api/batch/insert
```

Insert data into multiple databases in a single request:
```bash
curl -X POST https://${SERVER_NAME}/api/batch/insert \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{
    "mongodb_data": [
      {
        "collection": "users",
        "documents": [{"username": "user1"}]
      }
    ],
    "qdrant_data": [
      {
        "collection": "repository_embeddings",
        "points": [{"id": 1, "vector": [...]}]
      }
    ],
    "redis_data": [
      {"key": "cache:key", "value": "value"}
    ]
  }'
```

## Authentication

Most read operations are public. Write operations require an API key.

### Getting an API Key

Contact your system administrator to obtain a `DATA_INGESTION_API_KEY`.

### Using the API Key

Include the API key in the `X-API-Key` header:

```bash
curl -H "X-API-Key: YOUR_API_KEY" ...
```

## Database Access

### MongoDB Collections

Available collections in the `gitquery` database:
- `users` - User accounts and profiles
- `chat_sessions` - Chat interaction history
- `user_interactions` - User activity tracking
- `recommendations` - Repository recommendations

### Cosmos DB Collections

Available collections in the `gitquery_cosmos` database:
- `repositories` - GitHub repository metadata
- `repository_activity` - Repository activity and statistics

### Qdrant Collections

Available vector collections (768-dimensional embeddings):
- `repository_embeddings` - Repository code and description embeddings
- `code_embeddings` - Code snippet embeddings
- `user_preference_embeddings` - User preference vectors

### Redis Keys

Common key patterns:
- `session:{session_id}` - User session data
- `cache:{key}` - Cached query results
- `user:{user_id}:{data}` - User-specific data

## Data Ingestion Pipelines

### Python Pipeline Example

```python
import requests
import pandas as pd
from typing import List, Dict
import json

class GitQueryDataPipeline:
    """Data pipeline for ingesting data into Git-Query databases"""
    
    def __init__(self, api_base_url: str, api_key: str):
        self.api_base_url = api_base_url.rstrip('/')
        self.headers = {
            'Content-Type': 'application/json',
            'X-API-Key': api_key
        }
    
    def ingest_repositories(self, df: pd.DataFrame):
        """Ingest repository data from a pandas DataFrame"""
        documents = df.to_dict('records')
        
        # Insert into MongoDB
        response = requests.post(
            f'{self.api_base_url}/api/mongodb/insert',
            headers=self.headers,
            json={
                'database': 'gitquery_cosmos',
                'collection': 'repositories',
                'documents': documents
            }
        )
        return response.json()
    
    def ingest_embeddings(self, embeddings: List[Dict]):
        """Ingest vector embeddings into Qdrant"""
        response = requests.post(
            f'{self.api_base_url}/api/qdrant/insert',
            headers=self.headers,
            json={
                'collection': 'repository_embeddings',
                'points': embeddings
            }
        )
        return response.json()
    
    def cache_results(self, key: str, value: str, expire_seconds: int = 3600):
        """Cache query results in Redis"""
        response = requests.post(
            f'{self.api_base_url}/api/redis/set',
            headers=self.headers,
            json={
                'key': key,
                'value': value,
                'expire': expire_seconds
            }
        )
        return response.json()
    
    def batch_ingest(self, mongo_data=None, qdrant_data=None, redis_data=None):
        """Batch insert data into multiple databases"""
        payload = {}
        if mongo_data:
            payload['mongodb_data'] = mongo_data
        if qdrant_data:
            payload['qdrant_data'] = qdrant_data
        if redis_data:
            payload['redis_data'] = redis_data
        
        response = requests.post(
            f'{self.api_base_url}/api/batch/insert',
            headers=self.headers,
            json=payload
        )
        return response.json()


# Usage Example
if __name__ == '__main__':
    # Initialize pipeline
    pipeline = GitQueryDataPipeline(
        api_base_url='https://your-server.com',
        api_key='your-api-key'
    )
    
    # Load data
    df = pd.read_csv('repositories.csv')
    
    # Ingest repositories
    result = pipeline.ingest_repositories(df)
    print(f"Inserted {result['inserted_count']} repositories")
    
    # Ingest embeddings
    embeddings = [
        {
            'id': i,
            'vector': [0.1] * 768,  # Replace with actual embeddings
            'payload': {'repo_id': i, 'name': f'repo_{i}'}
        }
        for i in range(10)
    ]
    result = pipeline.ingest_embeddings(embeddings)
    print(f"Inserted {result['inserted_count']} embeddings")
```

### Jupyter Notebook Pipeline

```python
# Install dependencies
!pip install requests pandas numpy sentence-transformers

import requests
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

# Configuration
API_BASE_URL = 'https://your-server.com'
API_KEY = 'your-api-key'
HEADERS = {'Content-Type': 'application/json', 'X-API-Key': API_KEY}

# Load embedding model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Load your data
df = pd.read_csv('your_data.csv')

# Generate embeddings
texts = df['description'].tolist()
embeddings = model.encode(texts)

# Prepare data for insertion
points = [
    {
        'id': idx,
        'vector': embedding.tolist(),
        'payload': row.to_dict()
    }
    for idx, (embedding, (_, row)) in enumerate(zip(embeddings, df.iterrows()))
]

# Insert in batches
batch_size = 100
for i in range(0, len(points), batch_size):
    batch = points[i:i+batch_size]
    response = requests.post(
        f'{API_BASE_URL}/api/qdrant/insert',
        headers=HEADERS,
        json={'collection': 'repository_embeddings', 'points': batch}
    )
    print(f"Batch {i//batch_size + 1}: {response.json()}")
```

### Scheduled Data Ingestion

```python
import schedule
import time
from datetime import datetime

def ingest_daily_data():
    """Run daily data ingestion"""
    print(f"Starting daily ingestion at {datetime.now()}")
    
    pipeline = GitQueryDataPipeline(
        api_base_url='https://your-server.com',
        api_key='your-api-key'
    )
    
    # Fetch new data from your source
    df = fetch_new_repositories()  # Your data source
    
    # Ingest
    result = pipeline.ingest_repositories(df)
    print(f"Ingested {result['inserted_count']} records")

# Schedule daily at 2 AM
schedule.every().day.at("02:00").do(ingest_daily_data)

# Run scheduler
while True:
    schedule.run_pending()
    time.sleep(60)
```

## Python Examples

### Query Existing Data

```python
import requests

API_BASE_URL = 'https://your-server.com'

# Query MongoDB
response = requests.post(
    f'{API_BASE_URL}/api/mongodb/query',
    json={
        'collection': 'repositories',
        'filter': {'language': 'Python', 'stars': {'$gt': 1000}},
        'sort': {'stars': -1},
        'limit': 10
    }
)
repos = response.json()['documents']

# Search similar repositories using Qdrant
query_vector = [0.1] * 768  # Replace with actual embedding
response = requests.post(
    f'{API_BASE_URL}/api/qdrant/search',
    json={
        'collection': 'repository_embeddings',
        'vector': query_vector,
        'limit': 5
    }
)
similar_repos = response.json()['results']

# Get cached data from Redis
response = requests.get(f'{API_BASE_URL}/api/redis/get/popular_repos')
cached_data = response.json()['value']
```

### Data Analysis Example

```python
import pandas as pd
import requests

def fetch_all_repos(api_base_url: str, batch_size: int = 1000):
    """Fetch all repositories for analysis"""
    all_repos = []
    skip = 0
    
    while True:
        response = requests.post(
            f'{api_base_url}/api/mongodb/query',
            json={
                'database': 'gitquery_cosmos',
                'collection': 'repositories',
                'filter': {},
                'limit': batch_size,
                'skip': skip
            }
        )
        
        data = response.json()
        repos = data['documents']
        
        if not repos:
            break
        
        all_repos.extend(repos)
        skip += batch_size
    
    return pd.DataFrame(all_repos)

# Fetch and analyze
df = fetch_all_repos('https://your-server.com')
print(df.describe())
print(df.groupby('language')['stars'].mean())
```

## Best Practices

### 1. Batch Operations

Use batch endpoints for bulk operations to reduce network overhead:

```python
# Good: Batch insert
pipeline.batch_ingest(
    mongo_data=[...],
    qdrant_data=[...],
    redis_data=[...]
)

# Avoid: Multiple individual requests
for item in items:
    pipeline.ingest_single(item)  # Inefficient
```

### 2. Error Handling

Always implement retry logic and error handling:

```python
import time
from requests.exceptions import RequestException

def insert_with_retry(data, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
```

### 3. Data Validation

Validate data before insertion:

```python
from pydantic import BaseModel, validator

class Repository(BaseModel):
    repo_id: int
    full_name: str
    stars: int
    language: str
    
    @validator('stars')
    def stars_must_be_positive(cls, v):
        if v < 0:
            raise ValueError('stars must be positive')
        return v
```

### 4. Rate Limiting

Respect rate limits:

```python
import time
from functools import wraps

def rate_limit(calls_per_second=10):
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

@rate_limit(calls_per_second=5)
def insert_data(data):
    return pipeline.ingest_repositories(data)
```

### 5. Logging

Implement comprehensive logging:

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def ingest_data(data):
    logger.info(f"Starting ingestion of {len(data)} records")
    try:
        result = pipeline.ingest_repositories(data)
        logger.info(f"Successfully ingested {result['inserted_count']} records")
        return result
    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}", exc_info=True)
        raise
```

## Support

For questions or issues:
- API Documentation: `https://${SERVER_NAME}/docs`
- GitHub Issues: [Project Repository](https://github.com/Seraphim-Systems/git-query)
- Email: support@example.com

## API Reference

For complete API reference and interactive testing, visit the Swagger UI at:
`https://${SERVER_NAME}/docs`
