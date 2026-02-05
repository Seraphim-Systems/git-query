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
from src.gateway.routers import auth, chat, recommendations, user

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - initialize and cleanup resources."""
    logger.info("Starting API Gateway...")

    # Initialize Redis
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    app.state.redis = redis
    logger.info(f"Connected to Redis: {settings.redis_url}")

    # Initialize MongoDB
    mongo_client = AsyncIOMotorClient(settings.mongodb_url)
    app.state.mongodb = mongo_client[settings.mongodb_db]
    logger.info(f"Connected to MongoDB: {settings.mongodb_url}")

    # Initialize services
    app.state.session_manager = SessionManager(redis)
    app.state.user_service = UserService(app.state.mongodb, redis)
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
app.add_middleware(SessionMiddleware)
app.add_middleware(
    RateLimitMiddleware,
    max_requests=settings.rate_limit_requests,
    window_seconds=settings.rate_limit_window,
)


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


# Include routers
app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(chat.router, prefix="/chat", tags=["Chat"])
app.include_router(
    recommendations.router, prefix="/recommend", tags=["Recommendations"]
)
app.include_router(user.router, prefix="/user", tags=["User"])


# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "api-gateway"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
