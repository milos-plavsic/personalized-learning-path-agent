from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_health() -> None:
    assert client.get("/health").status_code == 200


def test_roadmap() -> None:
    r = client.post("/v1/roadmap", json={"goal": "Learn ML engineering"})
    assert r.status_code == 200
    assert r.json()["goal"] == "Learn ML engineering"
