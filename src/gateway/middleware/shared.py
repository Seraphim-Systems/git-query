"""Shared middleware helpers and constants for the gateway."""

# Prefixes of paths handled by the gateway itself (API routes, auth, health).
# Any request path that does NOT start with one of these is a frontend request
# and should be proxied to the web container without gateway auth.
GATEWAY_API_PREFIXES = (
    "/auth",
    "/chat",
    "/recommend",
    "/user",
    "/api",
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
)

# Public API endpoints that don't require authentication or an API key.
# Only paths within GATEWAY_API_PREFIXES need to be listed here.
PUBLIC_PATHS = [
    "/health",
    "/api/health",
    "/api/health/databases",
    "/favicon.ico",
    "/docs/",
    "/openapi.json",
    "/auth/login",
    "/auth/register",
    # Repo lookup reads only public data; no key needed from browser clients.
    "/api/repos/lookup",
]

# Session-authenticated paths: these are user-facing routes that use cookie
# sessions rather than service API keys.  The API-key middleware must skip
# these so the session middleware can handle authentication instead.
# Both canonical paths (/recommend) and /api/* aliases are listed so the
# frontend can call either form and still receive session-based auth.
SESSION_PATHS = [
    "/auth",  # Auth routes (login/register are public, logout needs session)
    "/chat",  # AI chat - requires session
    "/recommend",  # Repository search - requires session
    "/user",  # User profile/preferences - requires session
    "/api/recommend",  # Alias used by the browser frontend
    "/api/chat",  # Alias used by the browser frontend
]
