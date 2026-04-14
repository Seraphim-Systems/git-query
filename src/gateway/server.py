"""API Gateway server - main entry point."""

from contextlib import asynccontextmanager
from datetime import datetime
from argon2 import PasswordHasher
from fastapi import FastAPI, Request
from pymongo.errors import DuplicateKeyError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from motor.motor_asyncio import AsyncIOMotorClient

from starlette.exceptions import HTTPException as StarletteHTTPException

from src.shared.config import settings
from src.gateway.services.session_manager import SessionManager
from src.gateway.services.user_service import UserService
from src.gateway.middleware.rate_limit import RateLimitMiddleware
from src.gateway.middleware.api_key import APIKeyMiddleware
from src.gateway.middleware.session import SessionMiddleware
from src.gateway.routers import auth, chat, recommendations, user, health, repos
from src.gateway.routers import mlflow_proxy
from src.gateway.middleware.shared import GATEWAY_API_PREFIXES
from src.shared.logging_config import configure_logging


# Include DB routers directly in the Gateway so the Gateway serves DB endpoints
# (replacing the separate db-query-api service). These routers originate from
# the storage package and provide Mongo, Redis, Qdrant and batch routes.
from src.storage.routers import (
    mongodb_router,
    redis_router,
    qdrant_router,
)

from src.db.clients import (
    startup_db_clients,
    shutdown_db_clients,
    get_redis_client,
    get_mongo_client,
)

logger = configure_logging(service_name="gateway", log_level=settings.log_level)
password_hasher = PasswordHasher()


async def _seed_admin_user(user_service) -> None:
    """Create the admin seed user if WEB_ADMIN_EMAIL is configured and the user doesn't exist yet."""
    email = settings.web_admin_email
    if not email:
        return

    password = settings.web_admin_password or ""
    username = settings.web_admin_username
    password_hash = password_hasher.hash(password)

    # Keep exactly one seeded admin aligned with configured secrets.
    removed = await user_service.db.users.delete_many(
        {
            "is_admin": True,
            "email": {"$ne": email},
        }
    )
    if removed.deleted_count:
        logger.warning(
            "Removed %s stale admin account(s) not matching WEB_ADMIN_EMAIL=%s",
            removed.deleted_count,
            email,
        )

    existing = await user_service.get_user_by_email(email)
    if existing:
        await user_service.db.users.update_one(
            {"_id": existing["_id"]},
            {
                "$set": {
                    "username": username,
                    "password_hash": password_hash,
                    "is_admin": True,
                    "updated_at": datetime.utcnow(),
                }
            },
        )
        logger.info("Admin seed user refreshed: %s", email)
        return

    try:
        await user_service.create_user(
            user_id=email,
            email=email,
            username=username,
            password_hash=password_hash,
            is_admin=True,
        )
        logger.info("Admin seed user created: %s (%s)", email, username)
    except DuplicateKeyError:
        await user_service.db.users.update_one(
            {"email": email},
            {
                "$set": {
                    "username": username,
                    "password_hash": password_hash,
                    "is_admin": True,
                    "updated_at": datetime.utcnow(),
                }
            },
        )
        logger.info(
            "Admin seed user refreshed after duplicate key: %s (%s)", email, username
        )


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Application lifespan manager - initialize and cleanup resources."""
    logger.info("Starting API Gateway...")
    # Start shared DB clients (used by storage package and internal services).
    # This keeps a single source-of-truth for DB client lifecycle.

    await startup_db_clients()

    # Keep existing async clients for Gateway internals (session manager
    # expects `redis.asyncio.Redis` and some gateway code uses Motor for async
    # Mongo access). Also expose the synchronous/shared clients via
    # `app_instance.state.db_clients` for components that need them.
    redis_async = Redis.from_url(settings.redis_url, decode_responses=True)
    app_instance.state.redis = redis_async

    # Motor (async) for gateway usage
    mongo_client = AsyncIOMotorClient(settings.mongodb_url)
    app_instance.state.mongodb = mongo_client[settings.mongodb_db]

    # Expose the sync/shared clients (from db.clients) for storage routers
    app_instance.state.db_clients = {
        "sync_redis": get_redis_client(),
        "sync_mongo": get_mongo_client(),
    }

    # Initialize services
    app_instance.state.session_manager = SessionManager(
        redis_async, ttl=settings.session_ttl
    )
    app_instance.state.user_service = UserService(
        app_instance.state.mongodb, redis_async
    )
    logger.info("Services initialized (gateway async + shared sync clients)")

    # Seed admin user if configured (idempotent – skipped if user already exists)
    await _seed_admin_user(app_instance.state.user_service)

    yield

    # Cleanup
    logger.info("Shutting down API Gateway...")
    await redis_async.close()
    mongo_client.close()

    # Call shared shutdown for db clients
    try:
        await shutdown_db_clients()
    except Exception:
        logger.exception("Error shutting down shared DB clients")


# Create FastAPI app
app = FastAPI(
    title="Git-Query API Gateway",
    description="API Gateway with authentication, session management, and request routing",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["Set-Cookie"],
)

# Add custom middleware
# Enforce API key auth for non-public endpoints. Session middleware handles
# user-facing routes (/chat, /recommend, /user).
app.add_middleware(SessionMiddleware)
app.add_middleware(APIKeyMiddleware)
app.add_middleware(
    RateLimitMiddleware,
    max_requests=settings.rate_limit_requests,
    window_seconds=settings.rate_limit_window,
)


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(_request: Request, exc: Exception):
    """Global exception handler."""
    logger.error("Unhandled exception: %s", exc, exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Return simple JSON for HTTP errors and log details (including 404)."""
    # Log 404s as warnings with request context; log other HTTP errors at info level
    msg = (
        f"HTTP {exc.status_code} on {request.method} {request.url.path} - {exc.detail}"
    )
    if exc.status_code == 404:
        logger.warning(msg)
    else:
        logger.info(msg)

    # Provide clearer, structured responses for common cases
    if exc.status_code == 404:
        content = {
            "detail": "Route not found",
            "path": request.url.path,
        }
    elif exc.status_code == 401:
        content = {
            "detail": "Not authenticated",
            "reason": exc.detail or "Authentication required",
        }
    else:
        content = {"detail": exc.detail if exc.detail else "HTTP error"}

    # Preserve any headers the original exception carried (e.g., WWW-Authenticate)
    headers = getattr(exc, "headers", None)
    if headers:
        return JSONResponse(
            status_code=exc.status_code, content=content, headers=headers
        )

    return JSONResponse(status_code=exc.status_code, content=content)


# Include routers
app.include_router(health.router)
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(chat.router, prefix="/chat", tags=["Chat"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(
    recommendations.router, prefix="/recommend", tags=["Recommendations"]
)
app.include_router(
    recommendations.router, prefix="/api/recommend", tags=["Recommendations"]
)
app.include_router(user.router, prefix="/user", tags=["User"])
app.include_router(user.router, prefix="/api/user", tags=["User"])

# Mount the storage routers under `/api` so DB endpoints are served by the
# Gateway process itself (previously provided by the db-query-api service).
# Mount storage routers under `/api` so service paths are `/api/{service}/...`.
# This avoids duplicated segments like `/api/mongodb/mongodb` and removes the
# legacy `/api/db/...` namespace which previously produced `/api/db/mongodb/mongodb`.
app.include_router(mongodb_router.router, prefix="/api")
app.include_router(redis_router.router, prefix="/api")
app.include_router(qdrant_router.router, prefix="/api")
app.include_router(repos.router, prefix="/api")

# MLFlow UI proxy - admin-only, proxied to internal git-query-mlflow container
app.include_router(mlflow_proxy.router, prefix="/mlflow", tags=["MLFlow"])
app.include_router(mlflow_proxy.static_router, tags=["MLFlow"])


# Recommender admin proxy — forward /api/admin/models/* to the recommender service
@app.api_route(
    "/api/admin/models/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"]
)
async def proxy_recommender_admin(path: str, request: Request):
    """Proxy /api/admin/models/* to the recommender service admin endpoints."""
    import httpx
    from fastapi.responses import Response

    target = f"{settings.recommender_url}/admin/models/{path}"
    body = await request.body()
    forward_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.request(
                method=request.method,
                url=target,
                headers=forward_headers,
                params=request.query_params,
                content=body,
            )
        except httpx.ConnectError:
            logger.error(
                "Cannot connect to recommender at %s", settings.recommender_url
            )
            return Response(content=b"Recommender unavailable", status_code=503)
        except httpx.TimeoutException:
            logger.error("Timeout proxying to recommender admin at %s", target)
            return Response(content=b"Recommender timeout", status_code=504)

    excluded = {
        "transfer-encoding",
        "connection",
        "keep-alive",
        "te",
        "trailers",
        "upgrade",
    }
    resp_headers = {k: v for k, v in resp.headers.items() if k.lower() not in excluded}
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=resp_headers,
        media_type=resp.headers.get("content-type"),
    )


# Health check
@app.get("/health")
async def health_check(request: Request):
    """Health check endpoint."""
    # Delegate to the canonical API health handler so the top-level `/health`
    # returns the same detailed information as `/api/health`.
    from src.gateway.routers.health import health_check_all

    # Bridge the FastAPI Request to the delegated handler which expects one.
    # Import Request locally to avoid circular import issues during module
    # import time (the global Request type is already available).
    return await health_check_all(request)


# ── Frontend reverse proxy ────────────────────────────────────────────────────
# Catch-all: proxy any request that isn't a known API route to the web
# container. This makes port 80 the single entry point - no need to expose
# the web container's port 8080 externally.


@app.api_route("/{path:path}", methods=["GET", "HEAD", "OPTIONS"])
async def proxy_frontend(path: str, request: Request):
    """Reverse-proxy frontend assets and HTML pages to the web container."""
    import httpx
    from fastapi.responses import Response

    full_path = f"/{path}"

    # Don't intercept known gateway API routes (shouldn't normally reach here,
    # but guard in case of route ordering edge cases).
    if any(full_path.startswith(p) for p in GATEWAY_API_PREFIXES):
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="Route not found")

    web_url = settings.web_url
    target = f"{web_url}{full_path}"

    # Forward headers except host (rewritten by httpx to the target host)
    forward_headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }

    async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
        try:
            resp = await client.request(
                method=request.method,
                url=target,
                headers=forward_headers,
                params=request.query_params,
            )
        except httpx.ConnectError:
            logger.error("Cannot connect to web container at %s", web_url)
            return Response(content=b"Web frontend unavailable", status_code=503)
        except httpx.TimeoutException:
            logger.error("Timeout proxying to web container at %s", target)
            return Response(content=b"Web frontend timeout", status_code=504)

    # Strip hop-by-hop headers that mustn't be forwarded
    excluded = {
        "transfer-encoding",
        "connection",
        "keep-alive",
        "te",
        "trailers",
        "upgrade",
    }
    resp_headers = {k: v for k, v in resp.headers.items() if k.lower() not in excluded}

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=resp_headers,
        media_type=resp.headers.get("content-type"),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        log_config=None,
    )
