from fastapi.testclient import TestClient
from drone_traffic.api.main import create_app


def test_health_endpoint():
    app = create_app()
    client = TestClient(app)
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_api_docs():
    app = create_app()
    client = TestClient(app)
    response = client.get("/docs")
    assert response.status_code == 200
