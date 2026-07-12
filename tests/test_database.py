from backend.database import SessionLocal, init_db
from backend.models.alert_event import AlertLevel
from backend.schemas.alerts import AlertCreate
from backend.services.alert_service import create_alert


def test_create_alert():
    init_db()
    with SessionLocal() as db:
        alert = create_alert(
            db,
            AlertCreate(
                level=AlertLevel.ERROR,
                title="camera disconnected",
                summary="camera test",
                source_module="camera",
            ),
        )
        assert alert.id is not None
