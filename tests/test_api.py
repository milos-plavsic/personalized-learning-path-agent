from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_health() -> None:
    assert client.get("/health").status_code == 200


def test_roadmap_langgraph_loop() -> None:
    r = client.post(
        "/v1/roadmap",
        json={"goal": "improve graduation rates", "confidence_threshold": 0.75, "max_iterations": 2},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["goal"] == "improve graduation rates"
    assert "plan" in data
    assert 0.0 <= data["confidence_score"] <= 1.0
    assert 1 <= data["iterations"] <= 2
