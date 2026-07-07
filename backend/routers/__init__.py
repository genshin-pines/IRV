"""
API 路由注册
@owner 成员D (主) + 成员E (alerts 路由)
"""

from .alerts import router as alerts_router, broadcast_alert

__all__ = ["alerts_router", "broadcast_alert"]
