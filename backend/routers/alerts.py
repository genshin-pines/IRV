"""
告警 API 路由
@owner 成员E

端点:
  GET  /api/alerts        — 告警历史（分页 + 筛选）
  GET  /api/alerts/stats  — 告警统计
  GET  /api/alerts/{id}   — 告警详情
  POST /api/alerts/{id}/acknowledge — 确认告警
  GET  /api/logs          — 实时日志查询
  GET  /api/logs/stats    — 日志统计
  WS   /ws/alerts         — WebSocket 实时告警推送
"""

import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import JSONResponse
from sqlmodel import Session, create_engine

from backend.config import DATABASE_URL
from backend.models.alert_event import AlertEvent
from backend.services.alert_service import (
    get_alert_history, get_alert_stats, get_agent,
)
from backend.services.log_service import get_recent_logs, get_log_stats, get_collector
from backend.schemas.common import ok, fail

router = APIRouter(prefix="/api", tags=["告警 & 日志"])
logger = logging.getLogger(__name__)


def _err(message: str, status_code: int = 400, **kwargs) -> JSONResponse:
    """返回带 HTTP 状态码的统一错误响应"""
    return JSONResponse(status_code=status_code, content=fail(message, **kwargs))

# ── DB Session（临时，成员D 后续改为 FastAPI Depends 注入） ──
_engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})


def _get_session():
    with Session(_engine) as session:
        yield session


# ── 告警查询 ──────────────────────────────────────────

@router.get("/alerts")
def list_alerts(
    level: Optional[str] = Query(None, description="筛选级别: info|warning|critical"),
    hours: int = Query(24, ge=1, le=168, description="最近多少小时"),
    limit: int = Query(50, ge=1, le=500, description="返回条数上限"),
    session: Session = Depends(_get_session),
):
    """获取告警历史列表"""
    alerts = get_alert_history(session, hours=hours, level=level, limit=limit)
    return ok(data=[
        {
            "id": a.id,
            "level": a.level,
            "title": a.title,
            "summary": a.summary,
            "source_module": a.source_module,
            "ai_generated": a.ai_generated,
            "acknowledged": a.acknowledged,
            "created_at": a.created_at.isoformat(),
        }
        for a in alerts
    ])


@router.get("/alerts/stats")
def alert_stats(session: Session = Depends(_get_session)):
    """获取告警统计"""
    return ok(data=get_alert_stats(session))


@router.get("/alerts/{alert_id}")
def get_alert(alert_id: int, session: Session = Depends(_get_session)):
    """获取单条告警详情"""
    if alert_id <= 0:
        return _err(f"告警 ID 无效: {alert_id}", 400)
    alert = session.get(AlertEvent, alert_id)
    if not alert:
        return _err(f"告警 {alert_id} 不存在", 404)
    return ok(data={
        "id": alert.id,
        "level": alert.level,
        "title": alert.title,
        "summary": alert.summary,
        "detail": alert.detail,
        "source_module": alert.source_module,
        "affected_modules": alert.affected_modules.split(",") if alert.affected_modules else [],
        "ai_generated": alert.ai_generated,
        "webhook_markdown": alert.webhook_markdown,
        "acknowledged": alert.acknowledged,
        "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
        "created_at": alert.created_at.isoformat(),
    })


@router.post("/alerts/{alert_id}/acknowledge")
def acknowledge_alert(
    alert_id: int,
    acknowledged_by: str = "admin",
    session: Session = Depends(_get_session),
):
    """确认告警（标记为已处理）"""
    from datetime import datetime, timezone
    if alert_id <= 0:
        return _err(f"告警 ID 无效: {alert_id}", 400)
    alert = session.get(AlertEvent, alert_id)
    if not alert:
        return _err(f"告警 {alert_id} 不存在", 404)
    alert.acknowledged = True
    alert.acknowledged_at = datetime.now(timezone.utc)
    alert.acknowledged_by = acknowledged_by
    session.add(alert)
    session.commit()
    return ok(data={"alert_id": alert_id}, message="告警已确认")


# ── 日志查询 ──────────────────────────────────────────

@router.get("/logs")
def list_logs(n: int = Query(50, ge=1, le=500)):
    """获取最近 N 条系统日志"""
    logs = get_recent_logs(n)
    return ok(data=[
        {"seq": e.get("seq"), "timestamp": e["timestamp"],
         "module": e["module"], "level": e["level"],
         "message": e["message"]}
        for e in logs
    ])


@router.get("/logs/stats")
def log_stats():
    """获取日志统计"""
    return ok(data=get_log_stats())


# ── 调试：模拟异常日志（仅开发环境） ────────────────────

@router.post("/logs/simulate")
def simulate_anomaly_logs(
    scenario: str = Query("mixed", description="场景: error_spike|plate_low_conf|camera_disconnect|api_timeout|gesture_jitter|login_fail|mixed"),
    count: int = Query(15, ge=1, le=50),
):
    """
    注入模拟异常日志，用于测试实时告警监测。

    可用场景:
      - error_spike:      大量 ERROR 日志（触发错误率告警）
      - plate_low_conf:   车牌低置信度（触发识别告警）
      - camera_disconnect: 摄像头断连（触发严重告警）
      - api_timeout:      API 超时（触发性能告警）
      - gesture_jitter:   手势跳变（触发识别告警）
      - login_fail:       连续登录失败（触发安全告警）
      - mixed:            混合场景（多模块异常）
    """
    import logging

    collector = get_collector()

    SCENARIOS = {
        "error_spike": [
            ("plate_recognition", logging.ERROR, "RuntimeError: CUDA out of memory in plate_detect"),
            ("plate_recognition", logging.ERROR, "RuntimeError: ONNX inference failed, code=3"),
            ("plate_recognition", logging.ERROR, "ValueError: 图像解码失败: corrupt JPEG data"),
            ("api_server",       logging.ERROR, "Internal Server Error: POST /recognize 500"),
            ("api_server",       logging.ERROR, "ConnectionError: Redis 连接池耗尽"),
            ("database",         logging.ERROR, "OperationalError: (2003) Can't connect to MySQL"),
            ("api_server",       logging.ERROR, "ServiceUnavailable: 上游服务不可达"),
            ("camera_stream",    logging.ERROR, "RuntimeError: GStreamer pipeline 创建失败"),
        ],
        "plate_low_conf": [
            ("plate_recognition", logging.WARNING, "识别置信度偏低: 京A·HY35T, 置信度: 0.32, 耗时: 78ms"),
            ("plate_recognition", logging.WARNING, "识别置信度偏低: 沪B·8K219, 置信度: 0.41, 耗时: 85ms"),
            ("plate_recognition", logging.WARNING, "识别置信度偏低: 粤C·M7P4X, 置信度: 0.28, 耗时: 92ms"),
            ("plate_recognition", logging.WARNING, "识别置信度偏低: 川A·3F88T, 置信度: 0.19, 耗时: 80ms"),
        ],
        "camera_disconnect": [
            ("camera_stream", logging.ERROR, "RTSP 流中断: 桥面(东), 信号丢失, 正在重连..."),
            ("camera_stream", logging.ERROR, "RTSP 流中断: 隧道(事故段), 连接超时, 已重试3次"),
        ],
        "api_timeout": [
            ("api_server", logging.WARNING, "请求超时: POST /recognize-video, 耗时: 5200ms (阈值 500ms)"),
            ("api_server", logging.WARNING, "请求超时: GET /api/vehicles, 耗时: 4800ms"),
            ("api_server", logging.WARNING, "请求超时: POST /recognize, 耗时: 6100ms"),
            ("api_server", logging.WARNING, "请求超时: GET /dashboard/summary, 耗时: 3900ms"),
        ],
        "gesture_jitter": [
            ("gesture_recognition", logging.WARNING, "手势跳变频繁: 3帧内从'停止'切换到'直行'再切换到'左转'"),
            ("gesture_recognition", logging.WARNING, "手势跳变频繁: 2帧内切换5次, 置信度波动 > 0.4"),
            ("gesture_recognition", logging.WARNING, "手势跳变: 连续4帧识别结果不一致"),
        ],
        "login_fail": [
            ("auth", logging.WARNING, "登录失败: 用户 admin, 密码错误 (第1次尝试, IP: 192.168.1.50)"),
            ("auth", logging.WARNING, "登录失败: 用户 admin, 密码错误 (第2次尝试, IP: 192.168.1.50)"),
            ("auth", logging.WARNING, "登录失败: 用户 root, 密码错误 (第1次尝试, IP: 192.168.1.51)"),
            ("auth", logging.WARNING, "登录失败: 用户 admin, 密码错误 (第3次尝试, IP: 192.168.1.50)"),
            ("auth", logging.WARNING, "登录失败: 用户 admin, 密码错误 (第4次尝试, IP: 192.168.1.50)"),
        ],
    }

    VALID_SCENARIOS = {"error_spike", "plate_low_conf", "camera_disconnect",
                        "api_timeout", "gesture_jitter", "login_fail", "mixed"}

    if scenario not in VALID_SCENARIOS:
        return _err(f"未知场景: {scenario}，可选: {', '.join(sorted(VALID_SCENARIOS))}", 400)

    if scenario == "mixed":
        entries = []
        for s in ["plate_low_conf", "api_timeout", "camera_disconnect", "gesture_jitter", "login_fail"]:
            entries.extend(SCENARIOS[s])
        # 按顺序交错
        from itertools import cycle
        result = []
        for i in range(min(count, len(entries))):
            result.append(entries[i])
        entries = result
    else:
        entries = SCENARIOS[scenario]
        entries = entries[:count]

    for module, level, message in entries:
        mod_logger = logging.getLogger(module)
        mod_logger.log(level, message)

    return ok(data={
        "scenario": scenario,
        "injected": len(entries),
        "messages": [f"[{m}] [{l}] {msg[:60]}" for m, l, msg in entries],
    }, message="等待巡检周期结束后查看 GET /api/alerts")


# ── WebSocket 实时告警 ─────────────────────────────────

# 已连接的 WebSocket 客户端集合
_ws_clients: set[WebSocket] = set()


async def broadcast_alert(alert: dict):
    """向所有已连接的前端推送告警"""
    dead: list[WebSocket] = []
    for ws in _ws_clients:
        try:
            await ws.send_json(alert)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.discard(ws)


@router.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket):
    """WebSocket 端点 — 前端订阅实时告警推送"""
    await websocket.accept()
    _ws_clients.add(websocket)
    logger.info(f"WebSocket 客户端已连接 (当前 {len(_ws_clients)} 个)")
    try:
        while True:
            # 保持连接，等待客户端消息（心跳等）
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(websocket)
        logger.info(f"WebSocket 客户端已断开 (剩余 {len(_ws_clients)} 个)")


# ═══════════════════════════════════════════════════════════
# 感知事件接入（识别模块 → 事件总线）
# ═══════════════════════════════════════════════════════════

from pydantic import BaseModel
from typing import Any


class PerceptionEventInput(BaseModel):
    """识别模块推送的感知事件"""
    module: str  # "plate_recognition" | "traffic_gesture" | "driver_gesture"
    event_type: str = ""
    data: dict = {}
    confidence: float = 0.0
    camera_id: str = ""
    frame_timestamp: Optional[float] = None


@router.post("/perception/event")
async def ingest_perception_event(body: PerceptionEventInput):
    """
    接收来自识别模块的感知事件，发布到事件总线。

    车牌识别模块 (live_server.py) 每帧调用此端点，
    手势识别模块完成后也通过此端点推送事件。

    示例请求体:
    {
      "module": "plate_recognition",
      "event_type": "plate_detected",
      "data": {"code": "京A12345", "conf": 0.95, "color": "蓝牌", "bbox": [100,200,300,350]},
      "confidence": 0.95,
      "camera_id": "live1"
    }
    """
    from backend.services.alert_service import get_event_bus
    from fusion.perception_event import PerceptionEvent, Module, EventType

    bus = get_event_bus()
    if bus is None:
        return _err("EventBus 未初始化", 503)

    # 确定模块和事件类型
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

@router.get("/fusion/status")
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


@router.get("/latency/stats")
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

@router.post("/perception/simulate")
async def simulate_perception_events(
    scenario: str = Query("stop_with_vehicle", description="场景: stop_with_vehicle|slow_down|turn_left|multi_vehicle|normal|traffic_priority"),
):
    """
    模拟三路感知事件，用于测试融合推理引擎。

    可用场景:
      - stop_with_vehicle: 交警停止 + 前方有车 → 应建议停车
      - slow_down:        交警减速 + 前方有车 → 应建议减速
      - turn_left:        交警左转 + 前方有车 → 应建议注意观察
      - multi_vehicle:    多车检测 + 无交警 → 应建议注意车距
      - normal:           单模块检测，正常 → 应建议正常行驶
      - traffic_priority: 交警 + 车主手势冲突 → 应交警优先
    """
    from backend.services.alert_service import get_event_bus
    from fusion.perception_event import PerceptionEvent, Module, EventType

    bus = get_event_bus()
    if bus is None:
        return _err("EventBus 未初始化", 503)

    SCENARIOS = {
        "stop_with_vehicle": [
            # 交警停止手势
            {"module": "traffic_gesture", "event_type": "traffic_gesture",
             "data": {"gesture": "停止", "gesture_type": "stop", "confidence": 0.92},
             "confidence": 0.92, "camera_id": "live2"},
            # 车牌检测
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

    return ok(data={
        "scenario": scenario,
        "published": len(published_ids),
        "event_ids": published_ids,
    }, message="融合引擎将自动处理这些事件，查看 GET /api/fusion/status 获取结果")


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
