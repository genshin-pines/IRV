from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
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
from backend.schemas.common import ok, fail
from backend.services.alert_service import (
    acknowledge_alert as ack_alert_svc,
    delete_alert,
    get_alert,
    get_alert_stats,
    list_alerts,
)
from backend.services.log_service import get_log_stats, query_logs, write_log

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# WebSocket 管理器
# ═══════════════════════════════════════════════════════════

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


# ── 工具函数 ────────────────────────────────────────────

def _err(message: str, status_code: int = 400, **kwargs) -> JSONResponse:
    """返回带 HTTP 状态码的统一错误响应"""
    return JSONResponse(status_code=status_code, content=fail(message, **kwargs))


router = APIRouter(tags=["alerts"])


# ═══════════════════════════════════════════════════════════
# 告警 CRUD 路由
# ═══════════════════════════════════════════════════════════

@router.get("/api/alerts")
def api_list_alerts(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    level: str | None = Query(None),
    status_filter: str | None = Query(None, alias="status"),
    source_module: str | None = Query(None),
    db: Session = Depends(get_db),
):
    items, total = list_alerts(
        db, page=page, page_size=page_size,
        level=level, status=status_filter, source_module=source_module,
    )
    data = AlertList(
        items=[AlertRead.model_validate(item) for item in items],
        total=total, page=page, page_size=page_size,
    )
    return ok(data=data.model_dump())


@router.get("/api/alerts/stats")
def api_alert_stats(db: Session = Depends(get_db)):
    return ok(data=get_alert_stats(db))


@router.get("/api/alerts/{alert_id}")
def api_get_alert(alert_id: int, db: Session = Depends(get_db)):
    if alert_id <= 0:
        return _err(f"告警 ID 无效: {alert_id}", 400)
    alert = get_alert(db, alert_id)
    if alert is None:
        return _err(f"告警 {alert_id} 不存在", 404)
    return ok(data=AlertRead.model_validate(alert).model_dump())


@router.post("/api/alerts/{alert_id}/acknowledge")
def api_ack_alert(alert_id: int, payload: AlertAckRequest, db: Session = Depends(get_db)):
    if alert_id <= 0:
        return _err(f"告警 ID 无效: {alert_id}", 400)
    alert = ack_alert_svc(db, alert_id, payload.ack_user)
    if alert is None:
        return _err(f"告警 {alert_id} 不存在", 404)
    return ok(data=AlertRead.model_validate(alert).model_dump(), message="告警已确认")


@router.delete("/api/alerts/{alert_id}")
def api_delete_alert(alert_id: int, db: Session = Depends(get_db)):
    if alert_id <= 0:
        return _err(f"告警 ID 无效: {alert_id}", 400)
    if not delete_alert(db, alert_id):
        return _err(f"告警 {alert_id} 不存在", 404)
    return ok(data={"deleted": True})


# ═══════════════════════════════════════════════════════════
# 日志路由
# ═══════════════════════════════════════════════════════════

@router.get("/api/logs")
def api_list_logs(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    module: str | None = Query(None),
    level: str | None = Query(None),
    start_time: datetime | None = Query(None),
    end_time: datetime | None = Query(None),
):
    items, total = query_logs(
        page=page, page_size=page_size,
        module=module, level=level,
        start_time=start_time, end_time=end_time,
    )
    data = LogList(
        items=[LogRead(**item) for item in items],
        total=total, page=page, page_size=page_size,
    )
    return ok(data=data.model_dump())


@router.get("/api/logs/stats")
def api_log_stats():
    return ok(data=get_log_stats())


# ═══════════════════════════════════════════════════════════
# 模拟异常日志（开发测试用）
# ═══════════════════════════════════════════════════════════

@router.post("/api/logs/simulate")
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

    VALID_SCENARIOS = {"error_spike", "plate_low_conf", "camera_disconnect",
                        "api_timeout", "gesture_jitter", "login_fail", "mixed"}

    if payload.scenario not in VALID_SCENARIOS:
        return _err(f"未知场景: {payload.scenario}，可选: {', '.join(sorted(VALID_SCENARIOS))}", 400)

    if payload.scenario == "mixed":
        entries = []
        for key in ("plate_low_conf", "camera_disconnect", "api_timeout", "login_fail"):
            entries.extend(scenarios[key])
    else:
        entries = scenarios.get(payload.scenario)
        if entries is None:
            return _err(f"未知场景: {payload.scenario}", 400)

    written = 0
    for index in range(payload.count):
        source, level, message = entries[index % len(entries)]
        write_log(source, level, message)
        written += 1

    return ok(
        data={"scenario": payload.scenario, "injected": written},
        message="等待巡检周期结束后查看 GET /api/alerts",
    )


# ═══════════════════════════════════════════════════════════
# WebSocket 实时告警
# ═══════════════════════════════════════════════════════════

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
        logger.info(f"WebSocket 客户端已断开 (剩余 {len(ws_manager.clients)} 个)")


# ═══════════════════════════════════════════════════════════
# 感知事件接入（识别模块 → 事件总线）
# ═══════════════════════════════════════════════════════════


class PerceptionEventInput(BaseModel):
    """识别模块推送的感知事件"""
    module: str  # "plate_recognition" | "traffic_gesture" | "driver_gesture"
    event_type: str = ""
    data: dict = {}
    confidence: float = 0.0
    camera_id: str = ""
    frame_timestamp: Optional[float] = None


@router.post("/api/perception/event")
async def ingest_perception_event(body: PerceptionEventInput):
    """接收来自识别模块的感知事件，发布到事件总线。"""
    from backend.services.alert_service import get_event_bus
    from fusion.perception_event import PerceptionEvent, Module, EventType

    bus = get_event_bus()
    if bus is None:
        return _err("EventBus 未初始化", 503)

    # 确定模块
    try:
        module_enum = Module(body.module)
    except ValueError:
        return _err(f"未知模块: {body.module}，可选: plate_recognition/traffic_gesture/driver_gesture", 400)

    # 确定事件类型
    if body.event_type:
        try:
            event_type_enum = EventType(body.event_type)
        except ValueError:
            event_type_enum = {
                Module.PLATE_RECOGNITION: EventType.PLATE_DETECTED,
                Module.TRAFFIC_GESTURE: EventType.TRAFFIC_GESTURE,
                Module.DRIVER_GESTURE: EventType.DRIVER_GESTURE,
            }.get(module_enum, EventType.PLATE_DETECTED)
    else:
        event_type_enum = {
            Module.PLATE_RECOGNITION: EventType.PLATE_DETECTED,
            Module.TRAFFIC_GESTURE: EventType.TRAFFIC_GESTURE,
            Module.DRIVER_GESTURE: EventType.DRIVER_GESTURE,
        }.get(module_enum, EventType.PLATE_DETECTED)

    # 创建并发布事件
    event = PerceptionEvent(
        event_id=f"{body.module}_{int(time.time() * 1000)}",
        timestamp=datetime.now(timezone.utc),
        module=module_enum,
        event_type=event_type_enum,
        data=body.data,
        confidence=body.confidence,
        camera_id=body.camera_id,
        frame_timestamp=body.frame_timestamp or time.perf_counter(),
    )

    await bus.publish(event)
    return ok(data={"event_id": event.event_id})


# ═══════════════════════════════════════════════════════════
# 融合推理查询
# ═══════════════════════════════════════════════════════════

@router.get("/api/fusion/status")
async def fusion_status():
    """获取融合引擎状态和最新驾驶建议"""
    from backend.services.alert_service import get_fusion_agent, get_event_bus

    fusion = get_fusion_agent()
    bus = get_event_bus()

    result = {
        "fusion_agent": fusion.status if fusion else None,
        "event_bus": bus.stats if bus else None,
    }

    # 附加上下文快照
    if bus:
        try:
            context = await bus.get_context()
            result["context"] = {
                "plate": _simplify_context(context.get("plate", {})),
                "traffic_gesture": _simplify_context(context.get("traffic_gesture", {})),
                "driver_gesture": _simplify_context(context.get("driver_gesture", {})),
                "window_size": context.get("window_size", 0),
            }
        except Exception:
            result["context"] = None

    return ok(data=result)


@router.get("/api/latency/stats")
async def latency_stats():
    """获取全链路延迟统计"""
    from backend.services.alert_service import get_fusion_agent

    fusion = get_fusion_agent()
    if fusion is None:
        return _err("FusionAgent 未初始化", 503)
    return ok(data=fusion.latency_stats)


# ═══════════════════════════════════════════════════════════
# 模拟感知事件（开发测试用）
# ═══════════════════════════════════════════════════════════

@router.post("/api/perception/simulate")
async def simulate_perception_events(
    scenario: str = Query("stop_with_vehicle", description="场景: stop_with_vehicle|slow_down|turn_left|multi_vehicle|normal|traffic_priority"),
):
    """模拟三路感知事件，用于测试融合推理引擎。"""
    from backend.services.alert_service import get_event_bus
    from fusion.perception_event import PerceptionEvent, Module, EventType

    bus = get_event_bus()
    if bus is None:
        return _err("EventBus 未初始化", 503)

    SCENARIOS = {
        "stop_with_vehicle": [
            {"module": "traffic_gesture", "event_type": "traffic_gesture",
             "data": {"gesture": "停止", "gesture_type": "stop", "confidence": 0.92},
             "confidence": 0.92, "camera_id": "live2"},
            {"module": "plate_recognition", "event_type": "plate_detected",
             "data": {"plate_code": "京A12345", "plate_color": "蓝牌", "confidence": 0.95,
                       "bbox": [100, 200, 300, 350]},
             "confidence": 0.95, "camera_id": "live1"},
            {"module": "plate_recognition", "event_type": "plate_detected",
             "data": {"plate_code": "沪B67890", "plate_color": "绿牌", "confidence": 0.88,
                       "bbox": [400, 150, 600, 300]},
             "confidence": 0.88, "camera_id": "live1"},
        ],
        "slow_down": [
            {"module": "traffic_gesture", "event_type": "traffic_gesture",
             "data": {"gesture": "减速慢行", "gesture_type": "slow_down", "confidence": 0.90},
             "confidence": 0.90, "camera_id": "live2"},
            {"module": "plate_recognition", "event_type": "plate_detected",
             "data": {"plate_code": "粤C11111", "plate_color": "蓝牌", "confidence": 0.93,
                       "bbox": [200, 180, 400, 340]},
             "confidence": 0.93, "camera_id": "live1"},
        ],
        "turn_left": [
            {"module": "traffic_gesture", "event_type": "traffic_gesture",
             "data": {"gesture": "左转弯", "gesture_type": "turn_left", "confidence": 0.87},
             "confidence": 0.87, "camera_id": "live2"},
            {"module": "plate_recognition", "event_type": "plate_detected",
             "data": {"plate_code": "川A88888", "plate_color": "蓝牌", "confidence": 0.91,
                       "bbox": [50, 200, 250, 350]},
             "confidence": 0.91, "camera_id": "live1"},
        ],
        "multi_vehicle": [
            {"module": "plate_recognition", "event_type": "plate_detected",
             "data": {"plate_code": "京A11111", "plate_color": "蓝牌", "confidence": 0.95,
                       "bbox": [100, 200, 300, 350]},
             "confidence": 0.95, "camera_id": "live1"},
            {"module": "plate_recognition", "event_type": "plate_detected",
             "data": {"plate_code": "沪B22222", "plate_color": "绿牌", "confidence": 0.89,
                       "bbox": [400, 150, 600, 300]},
             "confidence": 0.89, "camera_id": "live1"},
            {"module": "plate_recognition", "event_type": "plate_detected",
             "data": {"plate_code": "粤C33333", "plate_color": "蓝牌", "confidence": 0.92,
                       "bbox": [700, 100, 900, 250]},
             "confidence": 0.92, "camera_id": "live1"},
        ],
        "normal": [
            {"module": "plate_recognition", "event_type": "plate_detected",
             "data": {"plate_code": "京A12345", "plate_color": "蓝牌", "confidence": 0.96,
                       "bbox": [300, 200, 500, 350]},
             "confidence": 0.96, "camera_id": "live1"},
        ],
        "traffic_priority": [
            {"module": "traffic_gesture", "event_type": "traffic_gesture",
             "data": {"gesture": "停止", "gesture_type": "stop", "confidence": 0.91},
             "confidence": 0.91, "camera_id": "live2"},
            {"module": "driver_gesture", "event_type": "driver_gesture",
             "data": {"gesture": "挥手", "gesture_type": "wave", "confidence": 0.85},
             "confidence": 0.85, "camera_id": "live3"},
        ],
    }

    PERCEPTION_SCENARIOS = set(SCENARIOS.keys())
    if scenario not in PERCEPTION_SCENARIOS:
        return _err(f"未知场景: {scenario}，可选: {', '.join(sorted(PERCEPTION_SCENARIOS))}", 400)

    events_data = SCENARIOS[scenario]
    published_ids = []

    for evt in events_data:
        try:
            module_enum = Module(evt["module"])
        except ValueError:
            continue
        try:
            event_type_enum = EventType(evt["event_type"])
        except ValueError:
            event_type_enum = EventType.PLATE_DETECTED

        event = PerceptionEvent(
            event_id=f"sim_{scenario}_{len(published_ids)}_{int(time.time()*1000)}",
            timestamp=datetime.now(timezone.utc),
            module=module_enum,
            event_type=event_type_enum,
            data=evt["data"],
            confidence=evt["confidence"],
            camera_id=evt.get("camera_id", ""),
            frame_timestamp=time.perf_counter(),
        )
        await bus.publish(event)
        published_ids.append(event.event_id)

    return ok(
        data={"scenario": scenario, "published": len(published_ids), "event_ids": published_ids},
        message="融合引擎将自动处理这些事件，查看 GET /api/fusion/status 获取结果",
    )


def _simplify_context(ctx: dict) -> dict:
    """简化上下文用于 API 返回（去掉完整事件对象，只保留摘要）"""
    latest = ctx.get("latest")
    return {
        "has_data": latest is not None,
        "latest_summary": (
            f"{latest.gesture_name or latest.plate_code}" if latest else None
        ),
        "latest_confidence": latest.confidence if latest else None,
        "count_2s": ctx.get("count_2s", 0),
        "avg_confidence": ctx.get("avg_confidence", 0),
        "stable_1s": ctx.get("stable_1s", False),
    }
