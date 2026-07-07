"""
日志服务 — 桥接 LogCollector 与 FastAPI
@owner 成员E

职责:
  1. 初始化 LogCollector 并注册到 Python logging 体系
  2. 提供查询接口（供 API router 调用）
"""

import logging
from typing import List, Dict

from alert_agent.logger import LogCollector

# 模块级单例
_collector: LogCollector | None = None


def setup_log_collector(capacity: int = 500) -> LogCollector:
    """
    初始化日志收集器并挂载到根 logger。

    调用一次（在 FastAPI startup 事件中）即可，
    之后所有模块的 logging 输出都会被自动收集。

    Args:
        capacity: 环形缓冲区容量

    Returns:
        LogCollector 实例
    """
    global _collector
    _collector = LogCollector(capacity=capacity)

    # 设置根 logger 格式（避免重复 handler）
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # 清除可能存在的默认 handler，避免重复输出
    if not any(isinstance(h, LogCollector) for h in root.handlers):
        root.addHandler(_collector)

    return _collector


def get_collector() -> LogCollector:
    """获取 LogCollector 实例"""
    if _collector is None:
        return setup_log_collector()
    return _collector


def get_recent_logs(n: int = 50) -> List[Dict]:
    """获取最近 N 条日志"""
    return get_collector().get_recent(n)


def get_log_stats() -> Dict:
    """获取日志统计"""
    return get_collector().get_stats()
