"""Shared middleware helpers and constants for the gateway."""

# Public endpoints that don't require authentication
PUBLIC_PATHS = [
    "/health",
    "/api/health",
    # Expose database health probes as public endpoints for liveness/readiness
    "/api/db/health",
    "/api/db/liveness",
    "/api/db/ready",
    "/docs",
    "/openapi.json",
    "/auth/login",
    "/auth/register",
]
