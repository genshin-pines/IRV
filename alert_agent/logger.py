"""
日志捕获工具

支持:
  - 监听 Python logging 模块的日志输出
  - 作为 log handler 收集日志并缓存，供 Agent 批量分析
  - 模拟日志生成（用于开发测试）

设计思路:
  各模块（车牌识别、手势识别、API 服务）使用 Python 标准 logging，
  本模块提供一个 LogCollector，作为 MemoryHandler 收集最近的日志条目，
  Agent 定时从 LogCollector 拉取新日志进行分析。
"""

import logging
import time
import threading
from collections import deque
from datetime import datetime, timezone
from typing import List, Dict


class LogCollector(logging.Handler):
    """
    日志收集器 — 同时是一个 logging.Handler 和一个环形缓冲区。

    用法:
        import logging
        from alert_agent.logger import LogCollector

        collector = LogCollector(capacity=500)
        logging.getLogger().addHandler(collector)

        # Agent 定时拉取
        new_logs = collector.get_new_logs()  # 返回上次拉取后的新日志
    """

    def __init__(self, capacity: int = 500, level: int = logging.INFO):
        """
        Args:
            capacity: 最多保留多少条日志
            level: 最低记录级别
        """
        super().__init__(level=level)
        self.capacity = capacity
        self._buffer: deque = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._last_read_seq: int = 0         # 上次拉取时最后一条的序列号
        self._seq: int = 0                   # 全局递增序列号
        self._total_emitted: int = 0

        # 设置格式
        self.setFormatter(logging.Formatter(
            "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))

    def emit(self, record: logging.LogRecord):
        """logging.Handler 接口：收到一条日志"""
        # 过滤 Agent 自身日志，避免循环分析
        if record.name.startswith("alert_agent"):
            return

        with self._lock:
            self._seq += 1
            self._total_emitted += 1
        entry = {
            "seq": self._seq,
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "module": record.name,
            "level": record.levelname,
            "message": self.format(record),
            "levelno": record.levelno,
        }
        with self._lock:
            self._buffer.append(entry)

    def get_new_logs(self) -> List[Dict]:
        """获取上次拉取之后新增的日志（基于序列号，缓冲区溢出也不会丢）"""
        with self._lock:
            items = list(self._buffer)
            if not items:
                return []
            # 找到所有 seq > _last_read_seq 的条目
            new_logs = [e for e in items if e["seq"] > self._last_read_seq]
            if new_logs:
                self._last_read_seq = new_logs[-1]["seq"]
            return new_logs

    def get_all_logs(self) -> List[Dict]:
        """获取当前缓冲区全部日志"""
        with self._lock:
            return list(self._buffer)

    def get_recent(self, n: int = 50) -> List[Dict]:
        """获取最近 n 条日志"""
        with self._lock:
            items = list(self._buffer)
            return items[-n:]

    def get_stats(self) -> Dict:
        """获取日志统计摘要"""
        with self._lock:
            items = list(self._buffer)
            if not items:
                return {"total": 0, "by_level": {}, "by_module": {}}

            by_level = {}
            by_module = {}
            for entry in items:
                lvl = entry["level"]
                mod = entry["module"]
                by_level[lvl] = by_level.get(lvl, 0) + 1
                by_module[mod] = by_module.get(mod, 0) + 1

            return {
                "total_emitted": self._total_emitted,
                "buffered": len(items),
                "by_level": by_level,
                "by_module": by_module,
            }


# ── 模拟日志生成（用于开发测试） ─────────────────────────────

def simulate_logs(collector: LogCollector, count: int = 20):
    """
    生成模拟日志，用于开发阶段测试 Agent 逻辑。

    模拟场景:
      - 正常车牌识别
      - 低置信度识别
      - API 超时
      - 摄像头断连
    """
    logger = logging.getLogger("simulator")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(collector)

    scenarios = [
        # (name, level, message)
        ("plate_recognition", logging.INFO,  "识别到车牌: 京A12345, 置信度: 0.95, 耗时: 68ms"),
        ("plate_recognition", logging.INFO,  "识别到车牌: 沪B67890, 置信度: 0.91, 耗时: 72ms"),
        ("plate_recognition", logging.INFO,  "识别到车牌: 粤C11111, 置信度: 0.88, 耗时: 65ms"),
        ("plate_recognition", logging.WARNING, "识别置信度偏低: 川D54321, 置信度: 0.42, 耗时: 89ms"),
        ("plate_recognition", logging.INFO,  "识别到车牌: 苏E99999, 置信度: 0.93, 耗时: 70ms"),
        ("plate_recognition", logging.WARNING, "识别置信度偏低: 浙F88888, 置信度: 0.38, 耗时: 85ms"),
        ("api_server",       logging.INFO,  "POST /recognize 200 120ms"),
        ("api_server",       logging.INFO,  "GET /docs 200 5ms"),
        ("api_server",       logging.WARNING, "请求超时: POST /recognize-video, 耗时: 3200ms"),
        ("api_server",       logging.INFO,  "POST /recognize 200 95ms"),
        ("gesture_recognition", logging.INFO,  "检测到手势: 停止 (交警), 置信度: 0.92"),
        ("gesture_recognition", logging.INFO,  "检测到手势: 直行 (交警), 置信度: 0.89"),
        ("gesture_recognition", logging.WARNING, "手势跳变频繁: 3帧内切换4次"),
        ("gesture_recognition", logging.INFO,  "检测到手势: 左转 (交警), 置信度: 0.87"),
        ("camera_stream",    logging.INFO,  "RTSP 流连接成功: 桥面 (10.126.59.120:8554/live1)"),
        ("camera_stream",    logging.WARNING, "帧率下降: 当前 12 FPS (正常 ≥25)"),
        ("camera_stream",    logging.ERROR, "RTSP 流中断: 隧道(事故), 正在重连..."),
        ("auth",             logging.INFO,  "用户登录成功: admin (192.168.1.100)"),
        ("auth",             logging.WARNING, "登录失败: 用户 user_03, 密码错误 (第3次尝试)"),
        ("database",         logging.INFO,  "MySQL 连接池状态: 活跃=3, 空闲=7, 最大=20"),
    ]

    for module, level, message in scenarios[:count]:
        # 使用 module 对应的 named logger
        mod_logger = logging.getLogger(module)
        mod_logger.setLevel(logging.DEBUG)
        mod_logger.addHandler(collector)
        mod_logger.log(level, message)
        time.sleep(0.05)  # 模拟时间间隔

    return collector.get_new_logs()


# ── 日志格式化输出工具 ───────────────────────────────────────

def format_logs_for_llm(logs: List[Dict], max_count: int = 100) -> str:
    """
    将日志列表格式化为适合发送给 LLM 的文本。

    Args:
        logs: 日志条目列表
        max_count: 最多包含多少条（避免 token 超限）

    Returns:
        格式化的文本块
    """
    if not logs:
        return "(暂无日志)"

    recent = logs[-max_count:]
    lines = []
    for entry in recent:
        lines.append(
            f"[{entry['timestamp']}] [{entry['module']}] "
            f"[{entry['level']}] {entry['message']}"
        )
    return "\n".join(lines)
