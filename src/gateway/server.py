"""API Gateway server - main entry point."""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from motor.motor_asyncio import AsyncIOMotorClient

from src.gateway.config import settings
from src.gateway.services.session_manager import SessionManager
from src.gateway.services.user_service import UserService
from src.gateway.middleware.session import SessionMiddleware
from src.gateway.middleware.rate_limit import RateLimitMiddleware
from src.gateway.middleware.api_key import APIKeyMiddleware
from src.gateway.routers import auth, chat, recommendations, user, health_v1, db_proxy

# Include DB routers directly in the Gateway so the Gateway serves DB endpoints
# (replacing the separate db-query-api service). These routers originate from
# the storage package and provide Mongo, Redis, Qdrant, Cosmos and batch routes.
from src.storage.routers import (
    mongodb_router,
    redis_router,
    qdrant_router,
    batch_router,
    cosmos_router,
)

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """Application lifespan manager - initialize and cleanup resources."""
    logger.info("Starting API Gateway...")

    # Initialize Redis
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    app_instance.state.redis = redis
    logger.info("Connected to Redis: %s", settings.redis_url)

    # Initialize MongoDB
    mongo_client = AsyncIOMotorClient(settings.mongodb_url)
    app_instance.state.mongodb = mongo_client[settings.mongodb_db]
    logger.info("Connected to MongoDB: %s", settings.mongodb_url)

    # Initialize services
    app_instance.state.session_manager = SessionManager(redis)
    app_instance.state.user_service = UserService(app_instance.state.mongodb, redis)
    logger.info("Services initialized")

    yield

    # Cleanup
    logger.info("Shutting down API Gateway...")
    await redis.close()
    mongo_client.close()


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
app.add_middleware(APIKeyMiddleware)  # API key auth for /api/db/*
app.add_middleware(SessionMiddleware)
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


from starlette.exceptions import HTTPException as StarletteHTTPException


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

    content = {"detail": exc.detail if exc.detail else "Not Found"}
    return JSONResponse(status_code=exc.status_code, content=content)


# Include routers
app.include_router(health_v1.router)  # /api/health
app.include_router(health_v1.db_router)  # /api/db/health
app.include_router(db_proxy.router)  # /api/db/*
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(chat.router, prefix="/chat", tags=["Chat"])
app.include_router(
    recommendations.router, prefix="/recommend", tags=["Recommendations"]
)
app.include_router(user.router, prefix="/user", tags=["User"])

# Mount the storage routers under `/api` so DB endpoints are served by the
# Gateway process itself (previously provided by the db-query-api service).
app.include_router(mongodb_router.router, prefix="/api")
app.include_router(redis_router.router, prefix="/api")
app.include_router(qdrant_router.router, prefix="/api")
app.include_router(batch_router.router, prefix="/api")
app.include_router(cosmos_router.router, prefix="/api")


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "api-gateway"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
