# ChatBots-GroupProject

## Request Flow & Routing

This repository contains two main HTTP surfaces related to database access:

- The API Gateway: central entry point for external clients and the web frontend.
	- External clients call routes under `/api` on the Gateway.
	- Database proxy routes are available at `/api/db/{service}/...` (for example
		`/api/db/mongodb/collections`). The Gateway applies middleware for API key
		validation, rate limiting, and session handling on these routes.

- The Database Query API (`db-query-api`): internal service that owns database
	clients (MongoDB, Redis, Qdrant, Cosmos, etc.) and performs actual database
	operations. Its routes are mounted under `/api/{service}/...` (for example
	`http://db-query-api:8080/api/mongodb/query`). This service should be
	reachable only from trusted internal networks.

Flows
-----
- External / web frontend:
	1. Client -> Gateway at `/api/db/{service}/...`.
	2. Gateway middleware validates request (API key, rate limits, sessions).
	3. Gateway forwards request to `db-query-api` at
		 `http://db-query-api:8080/api/{service}/...`.
	4. `db-query-api` uses its DB clients and returns the result back through
		 the Gateway to the client.

- Internal services / workers:
	1. Trusted internal services can call `db-query-api` directly at
		 `http://db-query-api:8080/api/{service}/...`.
	2. Direct calls bypass Gateway middleware; rely on deployment-level network
		 controls (internal network, mTLS, shared secrets) for security.

Security notes
--------------
- Keep database credentials and driver code inside `db-query-api` and avoid
	exposing them through the Gateway.
- Enforce per-service API keys and rate-limits at the Gateway for all
	external `/api/db/*` endpoints.
- Consider additional network-level protections for internal direct calls
	(namespace isolation, internal-only services, mTLS, etc.).
