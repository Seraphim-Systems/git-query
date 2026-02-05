"""Rate limiting middleware."""

import time
from collections import defaultdict
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiting middleware."""

    def __init__(self, app, max_requests: int = 100, window_seconds: int = 60):
        """
        Initialize rate limiter.

        Args:
            app: FastAPI application
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
        """
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_counts = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        """Process request and check rate limit."""

        # Get client identifier (IP or user_id if authenticated)
        client_id = request.client.host if request.client else "unknown"

        # If user is authenticated, use user_id for rate limiting
        if hasattr(request.state, "user_id"):
            client_id = request.state.user_id

        current_time = time.time()

        # Clean old requests outside the time window
        self.request_counts[client_id] = [
            req_time
            for req_time in self.request_counts[client_id]
            if current_time - req_time < self.window_seconds
        ]

        # Check if limit exceeded
        if len(self.request_counts[client_id]) >= self.max_requests:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Max {self.max_requests} requests per {self.window_seconds} seconds.",
            )

        # Add current request
        self.request_counts[client_id].append(current_time)

        # Process request
        response = await call_next(request)

        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(
            self.max_requests - len(self.request_counts[client_id])
        )
        response.headers["X-RateLimit-Reset"] = str(
            int(current_time + self.window_seconds)
        )

        return response
