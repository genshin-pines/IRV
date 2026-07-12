from fastapi.testclient import TestClient

from backend.main import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["ok"] is True


def test_simulate_logs():
    with TestClient(app) as client:
        response = client.post("/api/logs/simulate", json={"scenario": "camera_disconnect", "count": 1})
    assert response.status_code == 200
    assert response.json()["data"]["count"] == 1


def test_alert_lifecycle_and_stats():
    with TestClient(app) as client:
        client.post("/api/logs/simulate", json={"scenario": "camera_disconnect", "count": 10})
        response = client.get("/api/alerts")
        assert response.status_code == 200
        alert = response.json()["data"]["items"][0]

        detail = client.get(f"/api/alerts/{alert['id']}")
        assert detail.status_code == 200

        ack = client.post(f"/api/alerts/{alert['id']}/acknowledge", json={"ack_user": "tester"})
        assert ack.status_code == 200
        assert ack.json()["data"]["status"] == "ACKNOWLEDGED"

        stats = client.get("/api/alerts/stats")
        assert stats.status_code == 200
        assert "trend_4h" in stats.json()["data"]

        deleted = client.delete(f"/api/alerts/{alert['id']}")
        assert deleted.status_code == 200


def test_logs_filters_and_bad_simulation():
    with TestClient(app) as client:
        client.post("/api/logs/simulate", json={"scenario": "api_timeout", "count": 2})
        logs = client.get("/api/logs?module=backend&level=ERROR&page=1&page_size=5")
        assert logs.status_code == 200
        assert logs.json()["data"]["items"]

        bad = client.post("/api/logs/simulate", json={"scenario": "unknown", "count": 1})
        assert bad.status_code == 400
