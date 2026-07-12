import numpy as np
from fastapi.testclient import TestClient

from backend.main import app
from backend.services import mobile_camera_service as service


def test_validate_source_url_rejects_non_stream_url():
    try:
        service.validate_source_url("file:///tmp/phone.mp4")
    except ValueError as exc:
        assert "RTSP" in str(exc)
    else:
        raise AssertionError("local files must not be accepted as mobile streams")


def test_probe_and_connect_redact_stream_credentials(monkeypatch):
    class Capture:
        def __init__(self, *_args):
            pass

        def set(self, *_args):
            pass

        def isOpened(self):
            return True

        def read(self):
            return True, np.zeros((720, 1280, 3), dtype=np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(service.cv2, "VideoCapture", Capture)
    source = "rtsp://demo:secret@192.168.1.8:8554/phone"
    result = service.connect_source(source)
    assert result["width"] == 1280
    assert result["source_url"] == "rtsp://192.168.1.8:8554/phone"
    assert "secret" not in service.status()["source_url"]
    assert service.get_connected_source() == source
    service.disconnect_source()


def test_mobile_camera_api_status_and_disconnect():
    service.disconnect_source()
    with TestClient(app) as client:
        status = client.get("/api/mobile-camera/status")
        assert status.status_code == 200
        assert status.json()["data"]["configured"] is False

        disconnect = client.post("/api/mobile-camera/disconnect")
        assert disconnect.status_code == 200
        assert disconnect.json()["data"]["configured"] is False
