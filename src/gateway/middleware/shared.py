"""Shared middleware helpers and constants for the gateway."""

# Public endpoints that don't require authentication
PUBLIC_PATHS = [
    "/",
    "/health",
    "/api/health",
    "/api/health/databases",
    "/favicon.ico",
    "/docs/",
    "/openapi.json",
    "/auth/login",
    "/auth/register",
    # Frontend static assets and pages proxied to the web container
    "/login.html",
    "/register.html",
    "/home.html",
    "/styles/",
    "/src/",
]

# Session-authenticated paths: these are user-facing routes that use cookie
# sessions rather than service API keys.  The API-key middleware must skip
# these so the session middleware can handle authentication instead.
SESSION_PATHS = [
    "/auth",  # Auth routes (login/register are public, logout needs session)
    "/chat",  # AI chat - requires session
    "/recommend",  # Repository search - requires session
    "/user",  # User profile/preferences - requires session
]
