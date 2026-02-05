# Request Flow with Session & User Preferences

## Visual Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER BROWSER                                  │
│                                                                      │
│  [Login Form] → POST /api/auth/login                                │
│                        ↓                                             │
│                   {email, password}                                  │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     NGINX (Port 80)                                  │
│  • Receives all requests from frontend                              │
│  • Routes /api/* → api-gateway:8000                                 │
│  • Serves static files for /                                        │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────────┐
│                  API GATEWAY (Port 8000)                             │
│                                                                      │
│  ┌────────────────────────────────────────────────────┐            │
│  │  1. Route: /auth/login (PUBLIC)                    │            │
│  │     → Skip session check                           │            │
│  │     → Validate credentials against MongoDB         │            │
│  └────────────────────────┬───────────────────────────┘            │
│                           ↓                                         │
│  ┌────────────────────────────────────────────────────┐            │
│  │  2. Session Manager                                │            │
│  │     → Generate session_id = random_token()         │            │
│  │     → Store in Redis:                              │            │
│  │       session:{session_id} = {                     │            │
│  │         user_id: "user@example.com",               │            │
│  │         created_at: timestamp,                     │            │
│  │         ip: "192.168.1.1"                          │            │
│  │       }                                            │            │
│  │     → TTL: 24 hours                                │            │
│  └────────────────────────┬───────────────────────────┘            │
│                           ↓                                         │
│  ┌────────────────────────────────────────────────────┐            │
│  │  3. Response                                       │            │
│  │     → Set-Cookie: session_id={token}               │            │
│  │     → httpOnly, secure, samesite                   │            │
│  │     → Return: {user_id, username, message}         │            │
│  └────────────────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        USER BROWSER                                  │
│  • Session cookie stored automatically                              │
│  • All future requests include cookie                               │
│                                                                      │
│  [Get Recommendations] → GET /api/recommend?limit=10                │
│                             ↓                                        │
│                    Cookie: session_id={token}                       │
└────────────────────────┬────────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     NGINX (Port 80)                                  │
│  • Forwards to api-gateway with cookie                              │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────────┐
│                  API GATEWAY (Port 8000)                             │
│                                                                      │
│  ┌────────────────────────────────────────────────────┐            │
│  │  1. SessionMiddleware (BEFORE route handler)       │            │
│  │     → Extract session_id from cookie               │            │
│  │     → Validate in Redis                            │            │
│  │       ✓ Found: Continue                            │            │
│  │       ✗ Not found: Return 401 Unauthorized         │            │
│  └────────────────────────┬───────────────────────────┘            │
│                           ↓                                         │
│  ┌────────────────────────────────────────────────────┐            │
│  │  2. Load User Context                              │            │
│  │     → session.user_id = "user@example.com"         │            │
│  │     → Fetch from Redis cache:                      │            │
│  │       user_prefs:{user_id}                         │            │
│  │     → If not cached, fetch from MongoDB            │            │
│  │     → preferences = {                              │            │
│  │         languages: ["Python", "JavaScript"],       │            │
│  │         topics: ["ML", "web-dev"],                 │            │
│  │         min_stars: 100                             │            │
│  │       }                                            │            │
│  └────────────────────────┬───────────────────────────┘            │
│                           ↓                                         │
│  ┌────────────────────────────────────────────────────┐            │
│  │  3. Inject into Request State                      │            │
│  │     → request.state.user_id = "user@example.com"   │            │
│  │     → request.state.preferences = {...}            │            │
│  │     → request.state.session_id = token             │            │
│  └────────────────────────┬───────────────────────────┘            │
│                           ↓                                         │
│  ┌────────────────────────────────────────────────────┐            │
│  │  4. Route Handler: /recommend                      │            │
│  │     → User context already loaded!                 │            │
│  │     → Build payload for Recommender API:           │            │
│  │       {                                            │            │
│  │         user_id: "user@example.com",               │            │
│  │         preferences: {languages, topics...},       │            │
│  │         limit: 10                                  │            │
│  │       }                                            │            │
│  └────────────────────────┬───────────────────────────┘            │
│                           ↓                                         │
│  ┌────────────────────────────────────────────────────┐            │
│  │  5. HTTP Client Request                            │            │
│  │     → POST http://recommender-api:8002/recommend   │            │
│  │     → Body: {user_id, preferences, limit}          │            │
│  └────────────────────────────────────────────────────┘            │
└─────────────────────────┬───────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────────┐
│              RECOMMENDER API (Port 8002)                             │
│                                                                      │
│  ┌────────────────────────────────────────────────────┐            │
│  │  1. Receive Request with Full User Context         │            │
│  │     → user_id, preferences, limit                  │            │
│  └────────────────────────┬───────────────────────────┘            │
│                           ↓                                         │
│  ┌────────────────────────────────────────────────────┐            │
│  │  2. Load User Embedding from Qdrant               │            │
│  │     → collection: user_preference_embeddings       │            │
│  │     → user_vector = [0.1, 0.5, ..., 0.3]          │            │
│  └────────────────────────┬───────────────────────────┘            │
│                           ↓                                         │
│  ┌────────────────────────────────────────────────────┐            │
│  │  3. Query Qdrant with Filters                      │            │
│  │     → search(                                      │            │
│  │         query_vector=user_vector,                  │            │
│  │         filter={                                   │            │
│  │           "language": ["Python", "JavaScript"],    │            │
│  │           "stars": {"gte": 100},                   │            │
│  │           "archived": false                        │            │
│  │         },                                         │            │
│  │         limit=50                                   │            │
│  │       )                                            │            │
│  └────────────────────────┬───────────────────────────┘            │
│                           ↓                                         │
│  ┌────────────────────────────────────────────────────┐            │
│  │  4. Re-rank Results                                │            │
│  │     → Load user interactions from MongoDB          │            │
│  │     → Boost repos similar to clicked repos         │            │
│  │     → Apply business rules                         │            │
│  │     → Take top 10                                  │            │
│  └────────────────────────┬───────────────────────────┘            │
│                           ↓                                         │
│  ┌────────────────────────────────────────────────────┐            │
│  │  5. Return Recommendations                         │            │
│  │     → [                                            │            │
│  │         {repo_id, name, stars, description},       │            │
│  │         ...                                        │            │
│  │       ]                                            │            │
│  └────────────────────────────────────────────────────┘            │
└─────────────────────────┬───────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────────┐
│                  API GATEWAY (Port 8000)                             │
│                                                                      │
│  ┌────────────────────────────────────────────────────┐            │
│  │  6. Record Interaction                             │            │
│  │     → user_service.record_interaction(             │            │
│  │         user_id="user@example.com",                │            │
│  │         repo_id="recommendations_viewed",          │            │
│  │         action="view"                              │            │
│  │       )                                            │            │
│  │     → Store in MongoDB + Redis                     │            │
│  └────────────────────────┬───────────────────────────┘            │
│                           ↓                                         │
│  ┌────────────────────────────────────────────────────┐            │
│  │  7. Return Response                                │            │
│  │     → {                                            │            │
│  │         recommendations: [...],                    │            │
│  │         user_id: "user@example.com",               │            │
│  │         personalized: true                         │            │
│  │       }                                            │            │
│  └────────────────────────────────────────────────────┘            │
└─────────────────────────┬───────────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────────────┐
│                        USER BROWSER                                  │
│  • Displays personalized recommendations                            │
│  • User clicks on repo → POST /api/recommend/feedback               │
│  • Session automatically included                                   │
└─────────────────────────────────────────────────────────────────────┘


═══════════════════════════════════════════════════════════════════════

                            DATA STORES

┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│      Redis       │     │     MongoDB      │     │     Qdrant       │
├──────────────────┤     ├──────────────────┤     ├──────────────────┤
│                  │     │                  │     │                  │
│ • Sessions       │     │ • Users          │     │ • Repo vectors   │
│   TTL: 24h       │     │ • Preferences    │     │ • Code vectors   │
│                  │     │ • Interactions   │     │ • User vectors   │
│ • Pref cache     │     │ • Training data  │     │                  │
│   TTL: 1h        │     │                  │     │ Fast similarity  │
│                  │     │ Source of truth  │     │ search           │
│ • Rate limits    │     │                  │     │                  │
│                  │     │                  │     │                  │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

## Key Benefits

✅ **Single Sign-On**: Login once, use everywhere  
✅ **Automatic Context**: Every request has user data  
✅ **True Personalization**: Recommendations use preferences + history  
✅ **Security**: Internal services not exposed  
✅ **Performance**: Redis caching for fast lookups  
✅ **Scalability**: Gateway can scale independently  

## Why This Architecture?

1. **Frontend Never Authenticates Directly**: All auth through gateway
2. **User Context Everywhere**: MCP and Recommender get user data automatically
3. **Centralized Session Management**: One place to handle auth
4. **Cache Layer**: Fast preference lookups
5. **Interaction Tracking**: Every action recorded for ML training
6. **Clean Separation**: Frontend → Gateway → Services
