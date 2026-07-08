from __future__ import annotations

from datetime import datetime
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from sqlalchemy.orm import Session

from backend.database import get_db
from backend.schemas.alerts import (
    AlertAckRequest,
    AlertList,
    AlertRead,
    AlertStats,
    ApiResponse,
    LogList,
    LogRead,
    SimulateRequest,
)
from backend.services.alert_service import acknowledge_alert, delete_alert, get_alert, get_alert_stats, list_alerts
from backend.services.log_service import get_log_stats, query_logs, write_log


router = APIRouter(tags=["alerts"])


class WebSocketManager:
    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.clients.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self.clients.discard(websocket)

    async def broadcast(self, message: dict) -> None:
        closed = []
        for client in self.clients:
            try:
                await client.send_json(message)
            except Exception:
                closed.append(client)
        for client in closed:
            self.disconnect(client)


ws_manager = WebSocketManager()


def response(data=None, message: str = "success", ok: bool = True) -> dict:
    return {"ok": ok, "data": data, "message": message, "trace_id": datetime.now().strftime("%Y%m%d-") + uuid4().hex[:8]}


@router.get("/api/alerts", response_model=ApiResponse[AlertList])
def api_list_alerts(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    level: str | None = Query(None),
    status_filter: str | None = Query(None, alias="status"),
    source_module: str | None = Query(None),
    db: Session = Depends(get_db),
):
    items, total = list_alerts(db, page=page, page_size=page_size, level=level, status=status_filter, source_module=source_module)
    data = AlertList(items=[AlertRead.model_validate(item) for item in items], total=total, page=page, page_size=page_size)
    return response(data)


@router.get("/api/alerts/stats", response_model=ApiResponse[AlertStats])
def api_alert_stats(db: Session = Depends(get_db)):
    return response(get_alert_stats(db))


@router.get("/api/alerts/{alert_id}", response_model=ApiResponse[AlertRead])
def api_get_alert(alert_id: int, db: Session = Depends(get_db)):
    alert = get_alert(db, alert_id)
    if alert is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="alert not found")
    return response(AlertRead.model_validate(alert))


@router.post("/api/alerts/{alert_id}/acknowledge", response_model=ApiResponse[AlertRead])
def api_ack_alert(alert_id: int, payload: AlertAckRequest, db: Session = Depends(get_db)):
    alert = acknowledge_alert(db, alert_id, payload.ack_user)
    if alert is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="alert not found")
    return response(AlertRead.model_validate(alert))


@router.delete("/api/alerts/{alert_id}", response_model=ApiResponse[dict])
def api_delete_alert(alert_id: int, db: Session = Depends(get_db)):
    if not delete_alert(db, alert_id):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="alert not found")
    return response({"deleted": True})


@router.get("/api/logs", response_model=ApiResponse[LogList])
def api_list_logs(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    module: str | None = Query(None),
    level: str | None = Query(None),
    start_time: datetime | None = Query(None),
    end_time: datetime | None = Query(None),
):
    items, total = query_logs(page=page, page_size=page_size, module=module, level=level, start_time=start_time, end_time=end_time)
    data = LogList(items=[LogRead(**item) for item in items], total=total, page=page, page_size=page_size)
    return response(data)


@router.get("/api/logs/stats", response_model=ApiResponse[dict])
def api_log_stats():
    return response(get_log_stats())


@router.post("/api/logs/simulate", response_model=ApiResponse[dict])
def api_simulate_logs(payload: SimulateRequest):
    scenarios = {
        "plate_low_conf": [("plate", "WARNING", "plate confidence=0.62 low")],
        "camera_disconnect": [("camera", "ERROR", "RTSP disconnected Camera timeout")],
        "gesture_jitter": [
            ("gesture", "WARNING", "gesture result=left"),
            ("gesture", "WARNING", "gesture result=right"),
            ("gesture", "WARNING", "gesture result=stop"),
            ("gesture", "WARNING", "gesture result=left"),
            ("gesture", "WARNING", "gesture result=right"),
        ],
        "api_timeout": [("backend", "ERROR", "LLM request timeout elapsed=9s")],
        "login_fail": [("login", "WARNING", "login fail user=admin ip=192.168.1.50")],
    }
    if payload.scenario == "mixed":
        entries = []
        for key in ("plate_low_conf", "camera_disconnect", "api_timeout", "login_fail"):
            entries.extend(scenarios[key])
    else:
        entries = scenarios.get(payload.scenario)
        if entries is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="unknown scenario")

    written = 0
    for index in range(payload.count):
        source, level, message = entries[index % len(entries)]
        write_log(source, level, message)
        written += 1
    return response({"scenario": payload.scenario, "count": written})


@router.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            text = await websocket.receive_text()
            if text == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
