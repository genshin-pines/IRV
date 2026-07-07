"""
数据库模型
@owner 成员D (主, 补充 user / recognition_log 等表) + 成员E (alert_event)
"""

from .alert_event import AlertEvent

__all__ = ["AlertEvent"]
