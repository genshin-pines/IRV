from fastapi.testclient import TestClient

from backend.main import app


def test_websocket_ping():
    with TestClient(app) as client:
        with client.websocket_connect("/ws/alerts") as websocket:
            websocket.send_text("ping")
            assert websocket.receive_text() == "pong"


def test_websocket_multi_client_broadcast():
    with TestClient(app) as client:
        with client.websocket_connect("/ws/alerts") as ws1:
            with client.websocket_connect("/ws/alerts") as ws2:
                from backend.routers import ws_manager

                import anyio

                anyio.run(ws_manager.broadcast, {"title": "broadcast-test"})
                assert ws1.receive_json()["title"] == "broadcast-test"
                assert ws2.receive_json()["title"] == "broadcast-test"
