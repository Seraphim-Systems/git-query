from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.gateway.services.user_service import UserPreferences

import src.gateway.routers.chat as chat_router
import src.gateway.routers.recommendations as recommendations_router
import src.gateway.routers.user as user_router


class DummyPreferences:
    def __init__(self, payload=None):
        payload = payload or {
            "languages": ["Python"],
            "topics": ["llm"],
            "frameworks": [],
            "min_stars": 0,
            "max_age_days": 365,
            "exclude_archived": True,
            "exclude_forks": True,
            "license_types": [],
            "company_blacklist": [],
        }
        self.value = UserPreferences(**payload)

    def model_dump(self):
        return self.value.model_dump()


class DummyUserService:
    def __init__(self):
        self.updated_preferences_calls = []
        self.record_interaction_calls = []
        self.replace_user_chats_calls = []
        self.replace_saved_repos_calls = []
        self.replace_user_folders_calls = []
        self.interaction_history = [
            {"repo_id": "r1", "action": "click"},
            {"repo_id": "r2", "action": "save"},
        ]
        self.chats = [
            {
                "id": "chat-1",
                "title": "First session",
                "timestamp": 1710000000000,
                "messages": [
                    {"content": "hello", "role": "user", "timestamp": 1710000000000}
                ],
            }
        ]
        self.saved_repos = [
            {
                "id": "owner/repo",
                "name": "repo",
                "owner": "owner",
                "description": "desc",
                "stars": 1,
                "forks": 0,
                "language": "Python",
                "url": "https://github.com/owner/repo",
            }
        ]
        self.folders = [
            {
                "id": "folder-1",
                "name": "Pinned",
                "items": [{"id": "owner/repo", "name": "repo"}],
                "expanded": True,
            }
        ]
        self.user = {
            "_id": "mongo-id",
            "user_id": "user-123",
            "email": "user@example.com",
            "password_hash": "secret",
            "username": "irina",
        }

    async def get_user_preferences(self, user_id):
        return DummyPreferences({"languages": ["Python"], "topics": ["ai"]})

    async def update_preferences(self, user_id, preferences):
        self.updated_preferences_calls.append((user_id, preferences))
        return DummyPreferences(preferences)

    async def get_interaction_history(self, user_id, limit):
        return self.interaction_history[:limit]

    async def get_user_chats(self, user_id, limit):
        return self.chats[:limit]

    async def replace_user_chats(self, user_id, chats):
        self.replace_user_chats_calls.append((user_id, chats))
        self.chats = list(chats)
        return self.chats

    async def get_saved_repos(self, user_id, limit):
        return self.saved_repos[:limit]

    async def replace_saved_repos(self, user_id, repos):
        self.replace_saved_repos_calls.append((user_id, repos))
        self.saved_repos = list(repos)
        return self.saved_repos

    async def get_user_folders(self, user_id, limit):
        return self.folders[:limit]

    async def replace_user_folders(self, user_id, folders):
        self.replace_user_folders_calls.append((user_id, folders))
        self.folders = list(folders)
        return self.folders

    async def get_user(self, user_id):
        if self.user is None:
            return None
        return dict(self.user)

    async def record_interaction(
        self,
        user_id,
        repo_id,
        action,
        query="",
        variant="hybrid",
        position_in_results=None,
        metadata=None,
    ):
        self.record_interaction_calls.append(
            {
                "user_id": user_id,
                "repo_id": repo_id,
                "action": action,
                "query": query,
                "variant": variant,
                "position_in_results": position_in_results,
                "metadata": metadata,
            }
        )


class DummyHTTPResponse:
    def __init__(self, status_code=200, payload=None, should_raise=False):
        self.status_code = status_code
        self._payload = payload or {}
        self._should_raise = should_raise

    def raise_for_status(self):
        if self._should_raise:
            raise chat_router.httpx.HTTPError("upstream failed")

    def json(self):
        return self._payload


class DummyAsyncClient:
    next_post_response = DummyHTTPResponse(200, {})
    requests = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, url, json=None, timeout=None):
        self.__class__.requests.append({"url": url, "json": json, "timeout": timeout})
        return self.__class__.next_post_response


def override_current_user():
    return "user-123"


def override_user_preferences():
    return DummyPreferences()


def build_app():
    app = FastAPI()
    app.include_router(chat_router.router, prefix="/chat")
    app.include_router(recommendations_router.router, prefix="/recommend")
    app.include_router(user_router.router, prefix="/user")

    app.dependency_overrides[chat_router.get_current_user] = override_current_user
    app.dependency_overrides[chat_router.get_user_preferences] = (
        override_user_preferences
    )
    app.dependency_overrides[recommendations_router.get_current_user] = (
        override_current_user
    )
    app.dependency_overrides[recommendations_router.get_user_preferences] = (
        override_user_preferences
    )
    app.dependency_overrides[user_router.get_current_user] = override_current_user

    app.state.user_service = DummyUserService()
    return app


def test_chat_endpoint_success(monkeypatch):
    monkeypatch.setattr(chat_router.httpx, "AsyncClient", DummyAsyncClient)
    DummyAsyncClient.requests = []
    DummyAsyncClient.next_post_response = DummyHTTPResponse(
        200,
        {"response": "hello from mcp"},
    )

    app = build_app()
    client = TestClient(app)

    response = client.post(
        "/chat/",
        json={"message": "find me repos", "context": {"source": "ui"}},
    )

    assert response.status_code == 200
    assert response.json() == {
        "response": "hello from mcp",
        "user_id": "user-123",
    }
    assert len(DummyAsyncClient.requests) == 1
    sent = DummyAsyncClient.requests[0]["json"]
    assert sent["message"] == "find me repos"
    assert sent["user_id"] == "user-123"
    assert sent["context"] == {"source": "ui"}
    assert "preferences" in sent


def test_chat_endpoint_returns_fallback_message_on_http_error(monkeypatch):
    monkeypatch.setattr(chat_router.httpx, "AsyncClient", DummyAsyncClient)
    DummyAsyncClient.requests = []
    DummyAsyncClient.next_post_response = DummyHTTPResponse(
        500,
        {},
        should_raise=True,
    )

    app = build_app()
    client = TestClient(app)

    response = client.post("/chat/", json={"message": "hello", "context": {}})

    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == "user-123"
    assert "Error communicating with chat service" in body["response"]


def test_chat_endpoint_validates_missing_message():
    app = build_app()
    client = TestClient(app)

    response = client.post("/chat/", json={"context": {}})

    assert response.status_code == 422


def test_post_recommendations_success_uses_results_key(monkeypatch):
    monkeypatch.setattr(recommendations_router.httpx, "AsyncClient", DummyAsyncClient)
    DummyAsyncClient.requests = []
    DummyAsyncClient.next_post_response = DummyHTTPResponse(
        200,
        {
            "results": [{"repo_id": "r1"}, {"repo_id": "r2"}],
            "personalized": False,
        },
    )

    app = build_app()
    client = TestClient(app)

    response = client.post(
        "/recommend/",
        json={
            "query": "fastapi",
            "top_k": 5,
            "enable_personalization": True,
            "language": "Python",
            "min_stars": 100,
            "license": "MIT",
            "max_age_days": 30,
            "variant": "hybrid",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["user_id"] == "user-123"
    assert body["recommendations"] == [{"repo_id": "r1"}, {"repo_id": "r2"}]
    assert body["personalized"] is False

    sent = DummyAsyncClient.requests[0]["json"]
    assert sent["query"] == "fastapi"
    assert sent["top_k"] == 5
    assert sent["language"] == "Python"
    assert sent["min_stars"] == 100
    assert sent["license"] == "MIT"
    assert sent["max_age_days"] == 30
    assert sent["variant"] == "hybrid"


def test_post_recommendations_success_uses_items_key_when_results_missing(monkeypatch):
    monkeypatch.setattr(recommendations_router.httpx, "AsyncClient", DummyAsyncClient)
    DummyAsyncClient.requests = []
    DummyAsyncClient.next_post_response = DummyHTTPResponse(
        200,
        {
            "items": [{"repo_id": "x1"}],
            "personalized": True,
        },
    )

    app = build_app()
    client = TestClient(app)

    response = client.post("/recommend/", json={"query": "python"})

    assert response.status_code == 200
    body = response.json()
    assert body["recommendations"] == [{"repo_id": "x1"}]
    assert body["personalized"] is True


def test_post_recommendations_returns_empty_list_on_http_error(monkeypatch):
    monkeypatch.setattr(recommendations_router.httpx, "AsyncClient", DummyAsyncClient)
    DummyAsyncClient.requests = []
    DummyAsyncClient.next_post_response = DummyHTTPResponse(
        500,
        {},
        should_raise=True,
    )

    app = build_app()
    client = TestClient(app)

    response = client.post("/recommend/", json={"query": "broken"})

    assert response.status_code == 200
    body = response.json()
    assert body["recommendations"] == []
    assert body["user_id"] == "user-123"


def test_get_recommendations_success(monkeypatch):
    monkeypatch.setattr(recommendations_router.httpx, "AsyncClient", DummyAsyncClient)
    DummyAsyncClient.requests = []
    DummyAsyncClient.next_post_response = DummyHTTPResponse(
        200,
        {"results": [{"repo_id": "abc"}]},
    )

    app = build_app()
    client = TestClient(app)

    response = client.get("/recommend/?limit=3&query=llm")

    assert response.status_code == 200
    body = response.json()
    assert body["recommendations"] == [{"repo_id": "abc"}]
    assert body["personalized"] is True

    sent = DummyAsyncClient.requests[0]["json"]
    assert sent["top_k"] == 3
    assert sent["query"] == "llm"
    assert sent["user_id"] == "user-123"


def test_get_recommendations_returns_empty_list_on_http_error(monkeypatch):
    monkeypatch.setattr(recommendations_router.httpx, "AsyncClient", DummyAsyncClient)
    DummyAsyncClient.requests = []
    DummyAsyncClient.next_post_response = DummyHTTPResponse(
        500,
        {},
        should_raise=True,
    )

    app = build_app()
    client = TestClient(app)

    response = client.get("/recommend/?query=oops")

    assert response.status_code == 200
    body = response.json()
    assert body["recommendations"] == []
    assert body["user_id"] == "user-123"


def test_feedback_endpoint_records_interaction():
    app = build_app()
    client = TestClient(app)

    response = client.post(
        "/recommend/feedback",
        json={
            "repo_id": "repo-1",
            "action": "click",
            "query": "agents",
            "variant": "hybrid",
            "position_in_results": 2,
            "metadata": {"source": "test"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "status": "recorded",
        "repo_id": "repo-1",
        "action": "click",
        "variant": "hybrid",
    }

    calls = client.app.state.user_service.record_interaction_calls
    assert len(calls) == 1
    assert calls[0]["user_id"] == "user-123"
    assert calls[0]["repo_id"] == "repo-1"
    assert calls[0]["action"] == "click"
    assert calls[0]["position_in_results"] == 2
    assert calls[0]["metadata"] == {"source": "test"}


def test_get_user_preferences():
    app = build_app()
    client = TestClient(app)

    response = client.get("/user/preferences")

    assert response.status_code == 200
    body = response.json()
    assert "languages" in body
    assert "topics" in body
    assert "frameworks" in body
    assert "min_stars" in body
    assert "max_age_days" in body
    assert isinstance(body["languages"], list)
    assert isinstance(body["topics"], list)


def test_update_user_preferences():
    app = build_app()
    client = TestClient(app)

    payload = {
        "languages": ["Python", "Go"],
        "topics": ["search", "ranking"],
        "frameworks": ["FastAPI"],
        "min_stars": 50,
        "max_age_days": 90,
        "exclude_archived": True,
        "exclude_forks": True,
        "license_types": ["MIT"],
        "company_blacklist": ["foo"],
    }

    response = client.put("/user/preferences", json=payload)

    assert response.status_code == 200
    body = response.json()

    assert "languages" in body
    assert "topics" in body
    assert "frameworks" in body
    assert "min_stars" in body
    assert "max_age_days" in body

    calls = client.app.state.user_service.updated_preferences_calls
    assert len(calls) == 1
    assert calls[0][0] == "user-123"
    assert calls[0][1]["languages"] == ["Python", "Go"]
    assert calls[0][1]["topics"] == ["search", "ranking"]
    assert calls[0][1]["min_stars"] == 50


def test_get_interactions_returns_count():
    app = build_app()
    client = TestClient(app)

    response = client.get("/user/interactions?limit=1")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["interactions"] == [{"repo_id": "r1", "action": "click"}]


def test_get_user_chats_returns_count():
    app = build_app()
    client = TestClient(app)

    response = client.get("/user/chats?limit=10")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["chats"][0]["id"] == "chat-1"


def test_put_user_chats_replaces_collection():
    app = build_app()
    client = TestClient(app)

    payload = {
        "chats": [
            {
                "id": "chat-new",
                "title": "New Chat",
                "timestamp": 1710000001000,
                "messages": [
                    {"content": "yo", "role": "user", "timestamp": 1710000001000}
                ],
            }
        ]
    }

    response = client.put("/user/chats", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["chats"][0]["id"] == "chat-new"

    calls = client.app.state.user_service.replace_user_chats_calls
    assert len(calls) == 1
    assert calls[0][0] == "user-123"
    assert calls[0][1][0]["title"] == "New Chat"


def test_get_saved_repos_returns_count():
    app = build_app()
    client = TestClient(app)

    response = client.get("/user/saved-repos")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["repos"][0]["id"] == "owner/repo"


def test_put_saved_repos_replaces_collection():
    app = build_app()
    client = TestClient(app)

    payload = {
        "repos": [
            {
                "id": "foo/bar",
                "name": "bar",
                "owner": "foo",
                "description": "example",
                "stars": 10,
                "forks": 2,
                "language": "TypeScript",
                "url": "https://github.com/foo/bar",
            }
        ]
    }

    response = client.put("/user/saved-repos", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["repos"][0]["id"] == "foo/bar"

    calls = client.app.state.user_service.replace_saved_repos_calls
    assert len(calls) == 1
    assert calls[0][0] == "user-123"
    assert calls[0][1][0]["name"] == "bar"


def test_get_user_folders_returns_count():
    app = build_app()
    client = TestClient(app)

    response = client.get("/user/folders")

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["folders"][0]["id"] == "folder-1"


def test_put_user_folders_replaces_collection():
    app = build_app()
    client = TestClient(app)

    payload = {
        "folders": [
            {
                "id": "folder-2",
                "name": "Work",
                "items": [{"id": "chat-1", "title": "Session"}],
                "expanded": False,
            }
        ]
    }

    response = client.put("/user/folders", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 1
    assert body["folders"][0]["id"] == "folder-2"

    calls = client.app.state.user_service.replace_user_folders_calls
    assert len(calls) == 1
    assert calls[0][0] == "user-123"
    assert calls[0][1][0]["name"] == "Work"


def test_get_profile_returns_error_when_user_missing():
    app = build_app()
    app.state.user_service.user = None
    client = TestClient(app)

    response = client.get("/user/profile")

    assert response.status_code == 200
    assert response.json() == {"error": "User not found"}


def test_get_profile_returns_error_when_user_missing():
    app = build_app()
    app.state.user_service.user = None
    client = TestClient(app)

    response = client.get("/user/profile")

    assert response.status_code == 200
    assert response.json() == {"error": "User not found"}


def test_recommendations_missing_query_returns_200_with_fallback_shape():
    app = build_app()
    client = TestClient(app)

    response = client.post("/recommend/", json={})

    assert response.status_code == 200
    body = response.json()
    assert "recommendations" in body
    assert "personalized" in body
    assert "user_id" in body
    assert isinstance(body["recommendations"], list)


def test_user_profile_endpoint_exists():
    app = build_app()
    client = TestClient(app)

    response = client.get("/user/profile")

    assert response.status_code in (200, 422)


def test_unknown_route_returns_404():
    app = build_app()
    client = TestClient(app)

    response = client.get("/this-does-not-exist")

    assert response.status_code == 404
