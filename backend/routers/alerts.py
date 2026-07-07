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
from typing import Optional

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, Depends
from sqlmodel import Session, create_engine

from backend.config import DATABASE_URL
from backend.models.alert_event import AlertEvent
from backend.services.alert_service import (
    get_alert_history, get_alert_stats, get_agent,
)
from backend.services.log_service import get_recent_logs, get_log_stats, get_collector

router = APIRouter(prefix="/api", tags=["告警 & 日志"])
logger = logging.getLogger(__name__)

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
    limit: int = Query(50, ge=1, le=500),
    session: Session = Depends(_get_session),
) -> list[dict]:
    """获取告警历史列表"""
    alerts = get_alert_history(session, hours=hours, level=level, limit=limit)
    return [
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
    ]


@router.get("/alerts/stats")
def alert_stats(session: Session = Depends(_get_session)) -> dict:
    """获取告警统计"""
    return get_alert_stats(session)


@router.get("/alerts/{alert_id}")
def get_alert(alert_id: int, session: Session = Depends(_get_session)):
    """获取单条告警详情"""
    alert = session.get(AlertEvent, alert_id)
    if not alert:
        return {"error": "告警不存在"}, 404
    return {
        "id": alert.id,
        "level": alert.level,
        "title": alert.title,
        "summary": alert.summary,
        "detail": alert.detail,
        "source_module": alert.source_module,
        "affected_modules": alert.affected_modules.split(","),
        "ai_generated": alert.ai_generated,
        "webhook_markdown": alert.webhook_markdown,
        "acknowledged": alert.acknowledged,
        "acknowledged_at": alert.acknowledged_at.isoformat() if alert.acknowledged_at else None,
        "created_at": alert.created_at.isoformat(),
    }


@router.post("/alerts/{alert_id}/acknowledge")
def acknowledge_alert(
    alert_id: int,
    acknowledged_by: str = "admin",
    session: Session = Depends(_get_session),
):
    """确认告警（标记为已处理）"""
    from datetime import datetime, timezone
    alert = session.get(AlertEvent, alert_id)
    if not alert:
        return {"error": "告警不存在"}, 404
    alert.acknowledged = True
    alert.acknowledged_at = datetime.now(timezone.utc)
    alert.acknowledged_by = acknowledged_by
    session.add(alert)
    session.commit()
    return {"ok": True, "alert_id": alert_id}


# ── 日志查询 ──────────────────────────────────────────

@router.get("/logs")
def list_logs(n: int = Query(50, ge=1, le=500)):
    """获取最近 N 条系统日志"""
    logs = get_recent_logs(n)
    return [
        {"seq": e.get("seq"), "timestamp": e["timestamp"],
         "module": e["module"], "level": e["level"],
         "message": e["message"]}
        for e in logs
    ]


@router.get("/logs/stats")
def log_stats():
    """获取日志统计"""
    return get_log_stats()


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
        entries = SCENARIOS.get(scenario, SCENARIOS["error_spike"])
        entries = entries[:count]

    for module, level, message in entries:
        mod_logger = logging.getLogger(module)
        mod_logger.log(level, message)

    return {
        "ok": True,
        "scenario": scenario,
        "injected": len(entries),
        "messages": [f"[{m}] [{l}] {msg[:60]}" for m, l, msg in entries],
        "hint": "等待巡检周期结束后查看 GET /api/alerts",
    }


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
