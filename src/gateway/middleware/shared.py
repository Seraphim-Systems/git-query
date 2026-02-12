"""Shared middleware helpers and constants for the gateway."""

# Public endpoints that don't require authentication
PUBLIC_PATHS = [
    "/health",
    "/api/health",
    "/api/db/health",
    "/docs/",
    "/openapi.json",
    "/auth/login",
    "/auth/register",
]
