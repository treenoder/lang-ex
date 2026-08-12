from fastapi.testclient import TestClient

from tiny_llm.web.app import app


def test_home_and_empty_catalog() -> None:
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        response = client.get("/api/models")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
