# Request Flow & Session Management Architecture

## The Problem with Direct Frontend → MCP

❌ **Don't do this:**
```
Frontend → Nginx → MCP Server (no auth, no session, no user context)
```

**Why not?**
- No authentication/authorization
- No session tracking
- No user preference injection
- Security issues (exposing internal APIs)
- Can't personalize recommendations without user context

---

## ✅ Recommended Architecture: API Gateway Pattern

```
┌─────────────┐
│   Frontend  │
│  (Browser)  │
└──────┬──────┘
       │ HTTP Request + JWT/Cookie
       ↓
┌─────────────────────────────────────┐
│           NGINX (Port 80)           │
│  • SSL Termination                  │
│  • Static file serving              │
│  • Reverse proxy to API Gateway     │
└──────┬──────────────────────────────┘
       │
       ↓ /api/* → api-gateway:8000
┌─────────────────────────────────────┐
│        API Gateway Service          │
│     (Enhanced MCP Server)           │
│                                     │
│  ┌──────────────────────────────┐  │
│  │  Auth Middleware             │  │
│  │  • Validate JWT/session      │  │
│  │  • Rate limiting             │  │
│  └──────────┬───────────────────┘  │
│             ↓                       │
│  ┌──────────────────────────────┐  │
│  │  Session Manager             │  │
│  │  • Load session from Redis   │  │
│  │  • Refresh session TTL       │  │
│  └──────────┬───────────────────┘  │
│             ↓                       │
│  ┌──────────────────────────────┐  │
│  │  User Context Enrichment     │  │
│  │  • Fetch user profile        │  │
│  │  • Load preferences          │  │
│  │  • Get interaction history   │  │
│  └──────────┬───────────────────┘  │
│             ↓                       │
│  ┌──────────────────────────────┐  │
│  │  Request Router              │  │
│  │  • /chat → MCP tools         │  │
│  │  • /recommend → Reco API     │  │
│  │  • /user → User service      │  │
│  └──────────┬───────────────────┘  │
└─────────────┼───────────────────────┘
              │
        ┌─────┴──────┬──────────┐
        ↓            ↓          ↓
   ┌─────────┐  ┌─────────┐  ┌──────────┐
   │   MCP   │  │  Reco   │  │   User   │
   │  Tools  │  │  Engine │  │ Service  │
   └─────────┘  └─────────┘  └──────────┘
```

---

## Implementation: Enhanced MCP Server as API Gateway

### 1. Project Structure

```
src/
├── gateway/                    # NEW: API Gateway layer
│   ├── __init__.py
│   ├── server.py              # Main FastAPI app (orchestrator)
│   ├── config.py
│   │
│   ├── middleware/
│   │   ├── __init__.py
│   │   ├── auth.py            # JWT/session validation
│   │   ├── session.py         # Session management
│   │   ├── user_context.py    # Inject user data into request
│   │   └── rate_limit.py      # Request throttling
│   │
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── auth.py            # POST /login, /logout, /register
│   │   ├── chat.py            # POST /chat (proxies to MCP)
│   │   ├── recommendations.py # GET /recommend (proxies to Reco API)
│   │   └── user.py            # GET/PUT /user/preferences
│   │
│   └── services/
│       ├── __init__.py
│       ├── session_manager.py # Redis session operations
│       ├── user_service.py    # User CRUD + preferences
│       └── proxy.py           # HTTP client to backend services
│
├── mcp/                       # EXISTING: Keep as internal service
│   └── server.py              # MCP tool execution (not exposed directly)
│
└── recommender/               # EXISTING: Keep as internal service
    └── api/
        └── server.py          # Recommendation engine (not exposed directly)
```

---

## 2. Session Management Implementation

### Redis Session Schema

```python
# Redis keys
session:{session_id} = {
    "user_id": "user123",
    "created_at": 1738742400,
    "last_active": 1738742500,
    "ip_address": "192.168.1.1",
    "user_agent": "Mozilla/5.0..."
}
# TTL: 24 hours (auto-expires)

# User preference cache
user_prefs:{user_id} = {
    "languages": ["Python", "JavaScript"],
    "topics": ["machine-learning", "web-dev"],
    "min_stars": 100,
    "exclude_archived": true,
    "notification_preferences": {...}
}
# TTL: 1 hour (cache, fallback to MongoDB)
```

### Session Manager Service

```python
# src/gateway/services/session_manager.py

from typing import Optional
import json
import secrets
from datetime import datetime, timedelta
from redis.asyncio import Redis
from pydantic import BaseModel

class SessionData(BaseModel):
    user_id: str
    created_at: datetime
    last_active: datetime
    ip_address: str
    user_agent: str
    
class SessionManager:
    def __init__(self, redis: Redis):
        self.redis = redis
        self.session_ttl = 86400  # 24 hours
    
    async def create_session(
        self, 
        user_id: str, 
        ip: str, 
        user_agent: str
    ) -> str:
        """Create a new session and return session ID"""
        session_id = secrets.token_urlsafe(32)
        
        session_data = SessionData(
            user_id=user_id,
            created_at=datetime.utcnow(),
            last_active=datetime.utcnow(),
            ip_address=ip,
            user_agent=user_agent
        )
        
        await self.redis.setex(
            f"session:{session_id}",
            self.session_ttl,
            session_data.model_dump_json()
        )
        
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """Retrieve and refresh session"""
        key = f"session:{session_id}"
        data = await self.redis.get(key)
        
        if not data:
            return None
        
        session = SessionData.model_validate_json(data)
        
        # Update last_active and refresh TTL
        session.last_active = datetime.utcnow()
        await self.redis.setex(
            key,
            self.session_ttl,
            session.model_dump_json()
        )
        
        return session
    
    async def delete_session(self, session_id: str):
        """Delete session (logout)"""
        await self.redis.delete(f"session:{session_id}")
    
    async def get_user_sessions(self, user_id: str) -> list[str]:
        """Get all active sessions for a user"""
        pattern = "session:*"
        session_ids = []
        
        async for key in self.redis.scan_iter(match=pattern):
            data = await self.redis.get(key)
            if data:
                session = SessionData.model_validate_json(data)
                if session.user_id == user_id:
                    session_ids.append(key.decode().split(":")[-1])
        
        return session_ids
```

---

## 3. User Preference Service

### MongoDB User Schema

```python
# MongoDB: users collection
{
    "_id": ObjectId("..."),
    "user_id": "user123",
    "email": "user@example.com",
    "username": "johndoe",
    "created_at": ISODate("2025-01-01T00:00:00Z"),
    
    # User preferences for recommendations
    "preferences": {
        "languages": ["Python", "JavaScript", "Go"],
        "topics": ["machine-learning", "devops", "web-dev"],
        "frameworks": ["FastAPI", "React", "TensorFlow"],
        "min_stars": 100,
        "max_age_days": 365,
        "exclude_archived": true,
        "exclude_forks": true,
        "license_types": ["MIT", "Apache-2.0"],
        "company_blacklist": ["company1", "company2"]
    },
    
    # Interaction history (for collaborative filtering)
    "interaction_history": [
        {
            "repo_id": "repo123",
            "action": "star",  # star, view, clone, fork
            "timestamp": ISODate("2025-02-01T10:30:00Z"),
            "duration_seconds": 120
        }
    ],
    
    # Implicit signals
    "implicit_signals": {
        "frequently_viewed_repos": ["repo1", "repo2"],
        "last_search_queries": ["python async", "docker compose"],
        "clicked_recommendations": ["repo5", "repo6"]
    }
}
```

### User Service Implementation

```python
# src/gateway/services/user_service.py

from typing import Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorDatabase
from redis.asyncio import Redis
from pydantic import BaseModel
import json

class UserPreferences(BaseModel):
    languages: list[str] = []
    topics: list[str] = []
    frameworks: list[str] = []
    min_stars: int = 0
    max_age_days: int = 365
    exclude_archived: bool = True
    exclude_forks: bool = True
    license_types: list[str] = []
    company_blacklist: list[str] = []

class UserService:
    def __init__(self, mongodb: AsyncIOMotorDatabase, redis: Redis):
        self.db = mongodb
        self.redis = redis
        self.cache_ttl = 3600  # 1 hour
    
    async def get_user_preferences(self, user_id: str) -> UserPreferences:
        """Get user preferences with Redis cache"""
        # Check cache first
        cache_key = f"user_prefs:{user_id}"
        cached = await self.redis.get(cache_key)
        
        if cached:
            return UserPreferences.model_validate_json(cached)
        
        # Fetch from MongoDB
        user = await self.db.users.find_one({"user_id": user_id})
        
        if not user or "preferences" not in user:
            # Return default preferences
            prefs = UserPreferences()
        else:
            prefs = UserPreferences(**user["preferences"])
        
        # Cache for 1 hour
        await self.redis.setex(
            cache_key,
            self.cache_ttl,
            prefs.model_dump_json()
        )
        
        return prefs
    
    async def update_preferences(
        self, 
        user_id: str, 
        preferences: Dict[str, Any]
    ) -> UserPreferences:
        """Update user preferences"""
        # Update in MongoDB
        await self.db.users.update_one(
            {"user_id": user_id},
            {"$set": {"preferences": preferences}},
            upsert=True
        )
        
        # Invalidate cache
        await self.redis.delete(f"user_prefs:{user_id}")
        
        return UserPreferences(**preferences)
    
    async def record_interaction(
        self, 
        user_id: str, 
        repo_id: str, 
        action: str
    ):
        """Record user interaction for collaborative filtering"""
        interaction = {
            "repo_id": repo_id,
            "action": action,
            "timestamp": datetime.utcnow()
        }
        
        await self.db.users.update_one(
            {"user_id": user_id},
            {
                "$push": {
                    "interaction_history": {
                        "$each": [interaction],
                        "$slice": -1000  # Keep last 1000 interactions
                    }
                }
            },
            upsert=True
        )
        
        # Trigger async task to update user embedding
        from src.processing.tasks.embedding import update_user_embedding
        update_user_embedding.delay(user_id)
```

---

## 4. API Gateway Server Implementation

```python
# src/gateway/server.py

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis
from motor.motor_asyncio import AsyncIOMotorClient

from src.gateway.middleware.session import SessionMiddleware, get_current_user
from src.gateway.middleware.rate_limit import RateLimitMiddleware
from src.gateway.services.session_manager import SessionManager
from src.gateway.services.user_service import UserService
from src.gateway.routers import auth, chat, recommendations, user

# Global state
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize connections on startup"""
    # Redis connection
    redis = Redis.from_url("redis://redis:6379")
    app_state["redis"] = redis
    
    # MongoDB connection
    mongo_client = AsyncIOMotorClient("mongodb://mongodb:27017")
    app_state["mongodb"] = mongo_client.gitquery
    
    # Services
    app_state["session_manager"] = SessionManager(redis)
    app_state["user_service"] = UserService(app_state["mongodb"], redis)
    
    yield
    
    # Cleanup
    await redis.close()
    mongo_client.close()

# Create app
app = FastAPI(
    title="Git-Query API Gateway",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])
app.add_middleware(SessionMiddleware)
app.add_middleware(RateLimitMiddleware, max_requests=100, window_seconds=60)

# Routes
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(chat.router, prefix="/chat", tags=["Chat"])
app.include_router(recommendations.router, prefix="/recommend", tags=["Recommendations"])
app.include_router(user.router, prefix="/user", tags=["User"])

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

### Session Middleware

```python
# src/gateway/middleware/session.py

from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Optional

class SessionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Skip auth for public endpoints
        if request.url.path in ["/health", "/auth/login", "/auth/register"]:
            return await call_next(request)
        
        # Extract session from cookie or Authorization header
        session_id = request.cookies.get("session_id")
        if not session_id:
            auth_header = request.headers.get("Authorization")
            if auth_header and auth_header.startswith("Bearer "):
                session_id = auth_header.split(" ")[1]
        
        if not session_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No session found"
            )
        
        # Validate session
        session_manager = request.app.state.session_manager
        session = await session_manager.get_session(session_id)
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired session"
            )
        
        # Attach user context to request state
        request.state.user_id = session.user_id
        request.state.session_id = session_id
        
        # Fetch user preferences
        user_service = request.app.state.user_service
        preferences = await user_service.get_user_preferences(session.user_id)
        request.state.preferences = preferences
        
        response = await call_next(request)
        return response

# Dependency for route handlers
async def get_current_user(request: Request) -> str:
    """Dependency to get current user ID from request"""
    return request.state.user_id

async def get_user_preferences(request: Request):
    """Dependency to get user preferences"""
    return request.state.preferences
```

---

## 5. Router Implementations with User Context

### Recommendations Router (with preferences)

```python
# src/gateway/routers/recommendations.py

from fastapi import APIRouter, Depends, Request
from httpx import AsyncClient
from src.gateway.middleware.session import get_current_user, get_user_preferences

router = APIRouter()

@router.get("/")
async def get_recommendations(
    request: Request,
    user_id: str = Depends(get_current_user),
    preferences = Depends(get_user_preferences),
    limit: int = 10,
    context: str = None
):
    """
    Get personalized recommendations
    - User context is automatically injected from session
    - Preferences are used to filter/rank results
    """
    
    # Prepare request to Recommender API with user context
    payload = {
        "user_id": user_id,
        "preferences": preferences.model_dump(),
        "context": context,
        "limit": limit
    }
    
    # Call internal Recommender API
    async with AsyncClient() as client:
        response = await client.post(
            "http://recommender-api:8001/recommend",
            json=payload
        )
    
    recommendations = response.json()
    
    # Record implicit signal (viewed recommendations)
    user_service = request.app.state.user_service
    await user_service.record_interaction(
        user_id=user_id,
        repo_id="recommendations_viewed",
        action="view"
    )
    
    return recommendations

@router.post("/feedback")
async def record_feedback(
    request: Request,
    repo_id: str,
    action: str,  # click, star, clone, dismiss
    user_id: str = Depends(get_current_user)
):
    """Record user feedback on recommendations"""
    user_service = request.app.state.user_service
    await user_service.record_interaction(user_id, repo_id, action)
    
    # Trigger real-time model update (optional)
    from src.processing.tasks.embedding import update_user_embedding
    update_user_embedding.delay(user_id)
    
    return {"status": "recorded"}
```

### User Preferences Router

```python
# src/gateway/routers/user.py

from fastapi import APIRouter, Depends, Request
from src.gateway.middleware.session import get_current_user
from src.gateway.services.user_service import UserPreferences

router = APIRouter()

@router.get("/preferences")
async def get_preferences(
    request: Request,
    user_id: str = Depends(get_current_user)
):
    """Get current user preferences"""
    user_service = request.app.state.user_service
    prefs = await user_service.get_user_preferences(user_id)
    return prefs

@router.put("/preferences")
async def update_preferences(
    request: Request,
    preferences: UserPreferences,
    user_id: str = Depends(get_current_user)
):
    """Update user preferences"""
    user_service = request.app.state.user_service
    updated = await user_service.update_preferences(
        user_id, 
        preferences.model_dump()
    )
    return {"status": "updated", "preferences": updated}
```

---

## 6. Updated Nginx Configuration

```nginx
# web/nginx.conf

server {
    listen 80;
    server_name localhost;

    # All API requests go to API Gateway
    location /api/ {
        proxy_pass http://api-gateway:8000/;
        proxy_http_version 1.1;
        
        # Pass client info for session management
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Host $host;
        
        # WebSocket support (for real-time features)
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_cache_bypass $http_upgrade;
        
        # Timeouts
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }

    # Serve frontend static files
    location / {
        root /usr/share/nginx/html;
        index index.html;
        try_files $uri $uri/ /index.html;
    }

    # Health check
    location /health {
        access_log off;
        return 200 "healthy\n";
        add_header Content-Type text/plain;
    }
}
```

---

## 7. Frontend Request Flow

### Example: Get Personalized Recommendations

```javascript
// Frontend code (React/Vue/etc)

// 1. Login (get session)
const login = async (email, password) => {
  const response = await fetch('/api/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password }),
    credentials: 'include'  // Include cookies
  });
  
  const data = await response.json();
  // Session cookie is automatically set by browser
  return data;
};

// 2. Get recommendations (session is automatic)
const getRecommendations = async () => {
  const response = await fetch('/api/recommend?limit=20', {
    credentials: 'include'  // Send session cookie
  });
  
  // Backend automatically:
  // - Validates session
  // - Loads user preferences
  // - Applies personalization
  // - Returns filtered results
  
  return await response.json();
};

// 3. Record feedback
const recordFeedback = async (repoId, action) => {
  await fetch('/api/recommend/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ repo_id: repoId, action }),
    credentials: 'include'
  });
};

// 4. Update preferences
const updatePreferences = async (preferences) => {
  await fetch('/api/user/preferences', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(preferences),
    credentials: 'include'
  });
};
```

---

## 8. Updated Docker Compose

```yaml
services:
  # API Gateway (replaces direct MCP exposure)
  api-gateway:
    build:
      context: .
      dockerfile: infrastructure/docker/Dockerfile.gateway
    container_name: git-query-api-gateway
    expose:
      - "8000"
    environment:
      - GATEWAY_HOST=0.0.0.0
      - GATEWAY_PORT=8000
      - REDIS_URL=redis://:${REDIS_PASSWORD}@redis:6379
      - MONGODB_URL=mongodb://${MONGO_USER}:${MONGO_PASSWORD}@mongodb:27017/gitquery
      - MCP_SERVER_URL=http://mcp-server:8001
      - RECOMMENDER_URL=http://recommender-api:8002
    depends_on:
      - redis
      - mongodb
      - mcp-server
      - recommender-api
    networks:
      - git-query-app-network
      - git-query-db-network
    restart: unless-stopped

  # MCP Server (internal only, not exposed)
  mcp-server:
    build:
      context: .
      dockerfile: infrastructure/docker/Dockerfile.mcp
    container_name: git-query-mcp-server
    expose:
      - "8001"  # Changed port, not exposed externally
    environment:
      - MCP_HOST=0.0.0.0
      - MCP_PORT=8001
    networks:
      - git-query-app-network
    restart: unless-stopped

  # Recommender API (internal only)
  recommender-api:
    build:
      context: .
      dockerfile: infrastructure/docker/Dockerfile.recommender
    container_name: git-query-recommender-api
    expose:
      - "8002"  # Internal port
    environment:
      - API_HOST=0.0.0.0
      - API_PORT=8002
      - QDRANT_HOST=qdrant
    depends_on:
      - qdrant
      - redis
    networks:
      - git-query-app-network
      - git-query-db-network
    restart: unless-stopped

  # Nginx (only routes to API Gateway)
  nginx:
    build:
      context: .
      dockerfile: infrastructure/docker/Dockerfile.web
    container_name: git-query-nginx
    ports:
      - "80:80"
    depends_on:
      - api-gateway
    networks:
      - git-query-app-network
    restart: unless-stopped
```

---

## Summary: What You Get

✅ **Secure Authentication**: Session-based auth with Redis
✅ **User Context**: Every request has user ID + preferences automatically
✅ **Personalized Recommendations**: Preferences filter/rank results
✅ **Interaction Tracking**: Record user behavior for collaborative filtering
✅ **Cache Layer**: Redis caches preferences for fast access
✅ **Clean Separation**: Frontend never talks to internal services directly
✅ **Scalable**: API Gateway can scale independently

**Request Flow:**
```
Frontend → Nginx → API Gateway → [Session Check] → [Load Prefs] → Route to MCP/Reco
                                                                         ↓
                                                                    Return Results
```
