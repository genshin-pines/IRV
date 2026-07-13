"""
摄像头健康检查服务 — 周期性探测 RTSP 流连通性

检测到断联时写入 ERROR 日志 → CameraDisconnectRule 触发告警
检测到恢复时写入 INFO 日志  → 记录恢复事件

用法:
    from backend.services.camera_health_service import start_health_checker, stop_health_checker
    await start_health_checker()
    ...
    await stop_health_checker()
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict

import cv2

from backend.services.camera_service import CAMERAS, open_capture
from backend.services.log_service import write_log

logger = logging.getLogger("camera_health")


# ── 状态跟踪 ────────────────────────────────────────────

# 默认：连续 3 次探测失败才判定为真正断联
_DEFAULT_FAIL_THRESHOLD = 3


class CameraHealthState:
    """跟踪每个摄像头的健康状态，避免重复告警"""

    def __init__(self, fail_threshold: int = _DEFAULT_FAIL_THRESHOLD) -> None:
        self._fail_threshold = fail_threshold
        # camera_id → 是否在线（None=首次检测）
        self._online: Dict[str, bool | None] = {}
        # camera_id → 连续失败次数
        self._fail_count: Dict[str, int] = {}
        # camera_id → 首次失败时间
        self._first_fail_at: Dict[str, float] = {}

    def update(self, camera_id: str, is_online: bool) -> str | None:
        """
        更新摄像头状态，返回状态变化事件。
        返回 None 表示状态无变化，无需写日志。

        防抖策略：需连续 fail_threshold 次失败才判定为真正断联，
        避免网络瞬时抖动导致误报。
        """
        prev = self._online.get(camera_id)

        if is_online:
            self._fail_count[camera_id] = 0
            self._first_fail_at.pop(camera_id, None)
            self._online[camera_id] = True
            if prev is False:
                return "recovered"
            return None

        # 离线
        now = time.monotonic()
        self._fail_count[camera_id] = self._fail_count.get(camera_id, 0) + 1
        if camera_id not in self._first_fail_at:
            self._first_fail_at[camera_id] = now

        fail_count = self._fail_count[camera_id]

        # 未达到阈值：暂不判定为断联（可能是瞬时抖动）
        if fail_count < self._fail_threshold:
            return None

        # 刚好达到阈值：首次判定为真正断联
        if fail_count == self._fail_threshold and (prev is None or prev is True):
            self._online[camera_id] = False
            return "disconnected"

        # 已经离线，每 N 次（阈值整数倍）重复告警一次
        self._online[camera_id] = False
        if fail_count > self._fail_threshold and fail_count % self._fail_threshold == 0:
            return "still_disconnected"

        return None

    def snapshot(self) -> Dict[str, dict]:
        """返回所有摄像头的健康快照"""
        result = {}
        for cam in CAMERAS:
            cid = cam["id"]
            online = self._online.get(cid)
            result[cid] = {
                "name": cam["name"],
                "online": online,
                "fail_count": self._fail_count.get(cid, 0),
                "status": "online" if online else ("offline" if online is False else "unknown"),
            }
        return result


# ── 单摄像头探测 ─────────────────────────────────────────

async def _probe_camera(camera: dict, timeout_sec: float = 10.0, retries: int = 2) -> bool:
    """
    探测单个摄像头 RTSP 流是否可达。

    使用 cap.isOpened() + cap.read() 二阶验证：
    - isOpened() 可能返回 True 但实际流已断开（TCP 半开状态）
    - cap.read() 才能真正验证数据通路

    内置重试机制：单次探测可能因瞬时网络抖动失败，重试可过滤误报。

    Returns:
        True: 摄像头在线
        False: 摄像头离线
    """
    rtsp_url = camera["rtsp_url"]
    loop = asyncio.get_running_loop()

    def _probe() -> bool:
        cap = None
        try:
            cap = open_capture(rtsp_url)
            if not cap.isOpened():
                return False

            # 二阶验证：实际读取一帧
            ok, frame = cap.read()
            if not ok or frame is None:
                return False

            # 检查帧是否有效（非全黑/全灰的错误帧）
            if frame.size == 0:
                return False

            return True
        except Exception:
            return False
        finally:
            if cap is not None:
                cap.release()

    for attempt in range(retries + 1):
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(None, _probe), timeout=timeout_sec
            )
            if result:
                return True
            if attempt < retries:
                logger.debug(
                    "camera probe retry %d/%d: %s (%s)",
                    attempt + 1, retries, camera["id"], rtsp_url,
                )
                await asyncio.sleep(1.0)  # 重试前短暂等待
        except asyncio.TimeoutError:
            if attempt < retries:
                logger.debug(
                    "camera probe timeout retry %d/%d: %s (%s)",
                    attempt + 1, retries, camera["id"], rtsp_url,
                )
                await asyncio.sleep(1.0)
            else:
                logger.warning("camera probe timeout: %s (%s)", camera["id"], rtsp_url)
        except Exception:
            if attempt >= retries:
                logger.exception("camera probe error: %s", camera["id"])

    return False


# ── 活跃摄像头追踪 ───────────────────────────────────────
# 只有被前端查看的摄像头才参与健康检查，避免无关摄像头误报

_active_cameras: set[str] = set()
_active_lock = asyncio.Lock()


def mark_camera_active(camera_id: str) -> None:
    """标记摄像头正在被查看（MJPEG 流有观众时调用）"""
    _active_cameras.add(camera_id)


def mark_camera_inactive(camera_id: str) -> None:
    """标记摄像头不再被查看（所有观众断开时调用）"""
    _active_cameras.discard(camera_id)


def _get_active_cameras() -> list[dict]:
    """返回当前有观众在看的摄像头列表"""
    active_ids = set(_active_cameras)  # 快照
    return [cam for cam in CAMERAS if cam["id"] in active_ids]


# ── 健康检查调度 ─────────────────────────────────────────

_health_task: asyncio.Task | None = None
_health_state: CameraHealthState | None = None
_interval: float = 30.0


async def _health_loop(
    interval_sec: float = 30.0,
    fail_threshold: int = _DEFAULT_FAIL_THRESHOLD,
    probe_timeout_sec: float = 10.0,
) -> None:
    """后台健康检查循环（仅探测正在被查看的摄像头）"""
    global _health_state
    state = _health_state or CameraHealthState(fail_threshold=fail_threshold)
    _health_state = state

    logger.info(
        "camera health checker started interval=%.0fs fail_threshold=%d probe_max_sec=%.0fs",
        interval_sec, fail_threshold, probe_timeout_sec,
    )

    async def _probe_one(camera: dict) -> None:
        camera_id = camera["id"]
        camera_name = camera.get("name", camera_id)
        rtsp_url = camera.get("rtsp_url", "")

        is_online = await _probe_camera(camera, timeout_sec=probe_timeout_sec)
        change = state.update(camera_id, is_online)

        if change == "disconnected":
            write_log(
                "camera",
                "ERROR",
                f"Camera timeout url={rtsp_url} name={camera_name} "
                f"reason=health_check_failure",
            )
            logger.warning(
                "camera health check: %s (%s) DISCONNECTED",
                camera_name, camera_id,
            )
        elif change == "recovered":
            write_log(
                "camera",
                "INFO",
                f"Camera recovered url={rtsp_url} name={camera_name}",
            )
            logger.info(
                "camera health check: %s (%s) RECOVERED",
                camera_name, camera_id,
            )
        elif change == "still_disconnected":
            fail_count = state._fail_count[camera_id]
            write_log(
                "camera",
                "ERROR",
                f"Camera timeout url={rtsp_url} name={camera_name} "
                f"reason=health_check_failure fail_count={fail_count}",
            )
            logger.warning(
                "camera health check: %s (%s) still disconnected (fail_count=%d)",
                camera_name, camera_id, fail_count,
            )

    while True:
        try:
            active = _get_active_cameras()
            if active:
                tasks = [asyncio.create_task(_probe_one(cam)) for cam in active]
                await asyncio.gather(*tasks, return_exceptions=True)
            else:
                # 没有摄像头被查看，跳过本轮探测，但重置所有状态
                # 避免观众重连后立即因旧 fail_count 触发告警
                for cid in list(state._fail_count.keys()):
                    state._fail_count[cid] = 0
        except Exception:
            logger.exception("camera health check loop error")

        await asyncio.sleep(interval_sec)


async def start_health_checker(
    interval_sec: float = 30.0,
    fail_threshold: int = _DEFAULT_FAIL_THRESHOLD,
    probe_timeout_sec: float = 10.0,
) -> None:
    """启动摄像头健康检查后台任务"""
    global _health_task, _interval
    _interval = interval_sec

    if _health_task is not None and not _health_task.done():
        logger.warning("camera health checker already running")
        return

    _health_task = asyncio.create_task(
        _health_loop(interval_sec, fail_threshold=fail_threshold, probe_timeout_sec=probe_timeout_sec),
        name="camera-health-checker",
    )
    logger.info(
        "camera health checker task created interval=%.0fs fail_threshold=%d",
        interval_sec, fail_threshold,
    )


async def stop_health_checker() -> None:
    """停止摄像头健康检查"""
    global _health_task
    if _health_task is not None:
        _health_task.cancel()
        try:
            await _health_task
        except asyncio.CancelledError:
            pass
        _health_task = None
        logger.info("camera health checker stopped")


def get_health_state() -> CameraHealthState | None:
    """获取当前健康状态快照"""
    return _health_state


def health_status() -> dict:
    """返回健康检查服务状态"""
    state = _health_state
    return {
        "running": bool(_health_task and not _health_task.done()),
        "interval_seconds": _interval,
        "cameras": state.snapshot() if state else {},
    }
