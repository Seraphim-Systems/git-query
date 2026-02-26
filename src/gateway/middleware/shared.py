"""Shared middleware helpers and constants for the gateway."""

# Public endpoints that don't require authentication
PUBLIC_PATHS = [
    "/health",
    "/api/health",
    "/api/health/databases",
    "/favicon.ico",
    "/docs/",
    "/openapi.json",
    "/auth/login",
    "/auth/register",
]

# Session-authenticated paths: these are user-facing routes that use cookie
# sessions rather than service API keys.  The API-key middleware must skip
# these so the session middleware can handle authentication instead.
SESSION_PATHS = [
    "/chat",
    "/recommend",
    "/user",
]
