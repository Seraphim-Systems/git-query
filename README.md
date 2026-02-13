# ChatBots-GroupProject

## Request Flow & Routing

This repository exposes the API Gateway as the single HTTP surface for
external clients and the web frontend.

- The API Gateway: central entry point for external clients and the web frontend.
	- External clients call routes under `/api` on the Gateway.
	- Database routes are available at `/api/{service}/...` (for example
	  `/api/mongodb/collections`). The Gateway mounts the database routers
	  and applies middleware for API key validation, rate limiting, and session
	  handling on these routes.

Flows
-----
 - External / web frontend:
    1. Client -> Gateway at `/api/{service}/...`.
    2. Gateway middleware validates request (API key, rate limits, sessions).
    3. Gateway routes the request to the mounted DB routers which use local
	  DB client instances (running in the same process) or access internal
	  DB services as configured by your deployment.

 - Internal services / workers:
	 1. Internal services can call the Gateway's `/api/*` endpoints or, if you
		 run DB clients in separate internal services, call those internal
		 services directly depending on your deployment topology.
	 2. Ensure appropriate network-level protections (internal network, mTLS,
		 shared secrets) if bypassing the Gateway.

Security notes
--------------
- Keep database credentials and driver code inside internal services or the
  Gateway process and avoid exposing them directly to external clients.
- Enforce per-service API keys and rate-limits at the Gateway for all
	external `/api/db/*` endpoints.
- Consider additional network-level protections for internal direct calls
	(namespace isolation, internal-only services, mTLS, etc.).
