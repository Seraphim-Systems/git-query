"""Rate limiting middleware."""

import time
from collections import defaultdict
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from src.gateway.middleware.shared import PUBLIC_PATHS


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

        if self.max_requests <= 0 or self.window_seconds <= 0:
            return await call_next(request)

        path = request.url.path
        if path in PUBLIC_PATHS:
            return await call_next(request)

        forwarded_for = request.headers.get("x-forwarded-for", "")
        client_ip = forwarded_for.split(",", 1)[0].strip() if forwarded_for else None
        if not client_ip:
            client_ip = request.client.host if request.client else "unknown"

        now = time.time()
        window_start = now - self.window_seconds

        recent_requests = [
            ts for ts in self.request_counts[client_ip] if ts >= window_start
        ]

        if len(recent_requests) >= self.max_requests:
            retry_after = max(1, int(self.window_seconds - (now - recent_requests[0])))
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded"},
                headers={"Retry-After": str(retry_after)},
            )

        recent_requests.append(now)
        self.request_counts[client_ip] = recent_requests

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(
            max(0, self.max_requests - len(recent_requests))
        )
        response.headers["X-RateLimit-Reset"] = str(
            int(recent_requests[0] + self.window_seconds)
        )
        return response
