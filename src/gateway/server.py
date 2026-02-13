"""API Gateway server - main entry point."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
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
from src.gateway.routers import auth, chat, recommendations, user, health


# Include DB routers directly in the Gateway so the Gateway serves DB endpoints
# (replacing the separate db-query-api service). These routers originate from
# the storage package and provide Mongo, Redis, Qdrant, Cosmos and batch routes.
from src.storage.routers import (
    mongodb_router,
    redis_router,
    qdrant_router,
    cosmos_router,
)

from src.db.clients import (
    startup_db_clients,
    shutdown_db_clients,
    get_redis_client,
    get_mongo_client,
)

# Configure concise, human-friendly logging for container output
import sys
from datetime import datetime


class CompactFormatter(logging.Formatter):
    """Compact log formatter: ISO timestamp, single-letter level, short logger name.

    Examples:
      2026-02-13T15:04:05 I gateway: Started
      2026-02-13T15:04:06 W auth: Missing API key
    """

    LEVEL_MAP = {
        "DEBUG": "D",
        "INFO": "I",
        "WARNING": "W",
        "ERROR": "E",
        "CRITICAL": "C",
    }

    def formatTime(self, record, datefmt=None):
        return datetime.utcfromtimestamp(record.created).strftime("%Y-%m-%dT%H:%M:%S")

    def format(self, record):
        level = self.LEVEL_MAP.get(record.levelname, record.levelname[:1])
        short_name = record.name.split(".")[-1]
        time = self.formatTime(record)
        message = record.getMessage()

        if record.exc_info:
            # Append a one-line exception summary for brevity
            try:
                exc_text = self.formatException(record.exc_info).splitlines()[-1]
                message = f"{message} | {exc_text}"
            except Exception:
                pass

        return f"{time} {level} {short_name}: {message}"


# Attach compact formatter to root logger so all library logs (uvicorn, etc.) are concise
root_logger = logging.getLogger()
# Preserve any existing handlers in interactive/dev modes; replace basic config otherwise
if not root_logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(CompactFormatter())
    root_logger.addHandler(handler)
else:
    # Update existing handlers to use compact formatter
    for h in list(root_logger.handlers):
        try:
            h.setFormatter(CompactFormatter())
        except Exception:
            pass

root_logger.setLevel(settings.log_level)
logger = logging.getLogger(__name__)

# Quiet overly-verbose third-party loggers that flood container output
NOISY_LOGGERS = [
    "pymongo",
    "pymongo.topology",
    "pymongo.pool",
    "motor",
    "motor.motor_asyncio",
    "urllib3",
    "asyncio",
    "qdrant_client",
]

for name in NOISY_LOGGERS:
    try:
        logging.getLogger(name).setLevel(logging.WARNING)
    except Exception:
        pass


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
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
# Enforce API key auth for non-public endpoints. Session-based auth removed
# in favor of API-key-only access for all guarded routes.
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
app.include_router(chat.router, prefix="/chat", tags=["Chat"])
app.include_router(
    recommendations.router, prefix="/recommend", tags=["Recommendations"]
)
app.include_router(user.router, prefix="/user", tags=["User"])

# Mount the storage routers under `/api` so DB endpoints are served by the
# Gateway process itself (previously provided by the db-query-api service).
# Mount storage routers under `/api` so service paths are `/api/{service}/...`.
# This avoids duplicated segments like `/api/mongodb/mongodb` and removes the
# legacy `/api/db/...` namespace which previously produced `/api/db/mongodb/mongodb`.
app.include_router(mongodb_router.router, prefix="/api")
app.include_router(redis_router.router, prefix="/api")
app.include_router(qdrant_router.router, prefix="/api")
app.include_router(cosmos_router.router, prefix="/api")


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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
