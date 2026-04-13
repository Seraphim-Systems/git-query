from fastapi.testclient import TestClient

from tests.gateway.test_gateway_routers import build_app


def test_unknown_get_route_returns_404():
    app = build_app()
    client = TestClient(app)

    response = client.get("/this-route-does-not-exist")

    assert response.status_code == 404


def test_unknown_post_route_returns_404():
    app = build_app()
    client = TestClient(app)

    response = client.post("/unknown-endpoint", json={})

    assert response.status_code == 404


def test_wrong_method_on_chat_returns_405_or_404():
    app = build_app()
    client = TestClient(app)

    response = client.put("/chat/", json={"message": "hi"})

    assert response.status_code in (404, 405)


def test_wrong_method_on_user_preferences():
    app = build_app()
    client = TestClient(app)

    response = client.delete("/user/preferences")

    assert response.status_code in (404, 405)


def test_empty_json_body_on_chat():
    app = build_app()
    client = TestClient(app)

    response = client.post("/chat/", json={})

    assert response.status_code in (200, 422)


def test_empty_json_body_on_recommend():
    app = build_app()
    client = TestClient(app)

    response = client.post("/recommend/", json={})

    assert response.status_code == 200


def test_large_payload_on_recommend():
    app = build_app()
    client = TestClient(app)

    large_query = "python " * 1000

    response = client.post("/recommend/", json={"query": large_query})

    assert response.status_code == 200


def test_query_params_on_recommend_get():
    app = build_app()
    client = TestClient(app)

    response = client.get("/recommend/?limit=5&query=test")

    assert response.status_code == 200
    body = response.json()

    assert "recommendations" in body


def test_user_profile_basic_response():
    app = build_app()
    client = TestClient(app)

    response = client.get("/user/profile")

    assert response.status_code == 200
    body = response.json()

    assert "user_id" in body or "error" in body


def test_user_interactions_endpoint_exists():
    app = build_app()
    client = TestClient(app)

    response = client.get("/user/interactions")

    assert response.status_code == 200
    body = response.json()

    assert "interactions" in body
    assert "count" in body