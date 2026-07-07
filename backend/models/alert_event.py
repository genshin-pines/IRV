"""
数据库模型 — 告警事件表
@owner 成员E
"""

from datetime import datetime, timezone
from typing import Optional
from sqlmodel import SQLModel, Field


class AlertEvent(SQLModel, table=True):
    """告警事件 — 每次 Agent 发布告警时写入"""
    __tablename__ = "alert_events"

    id: Optional[int] = Field(default=None, primary_key=True)
    # 告警基本信息
    level: str = Field(index=True)               # info | warning | critical
    title: str                                   # 告警标题
    summary: str = Field(default="")             # 一句话摘要
    detail: str = Field(default="")              # 详细描述
    # 来源
    source_module: str = Field(default="")         # 触发模块
    affected_modules: str = Field(default="")     # 受影响模块(逗号分隔)
    # AI 相关信息
    ai_generated: bool = Field(default=False)    # 是否 LLM 生成
    fingerprint: str = Field(default="", index=True)  # 去重指纹
    # 通知
    webhook_markdown: str = Field(default="")     # Webhook 消息体
    # 状态
    acknowledged: bool = Field(default=False)     # 是否已确认
    acknowledged_at: Optional[datetime] = Field(default=None)
    acknowledged_by: str = Field(default="")
    # 时间
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        index=True,
    )
    ttl_minutes: int = Field(default=60)         # 告警有效时长

    class Config:
        json_schema_extra = {
            "example": {
                "level": "warning",
                "title": "车牌识别置信度持续偏低",
                "summary": "连续3帧置信度低于0.5",
                "detail": "plate_recognition 模块在过去5分钟内出现3次低置信度识别",
                "source_module": "plate_recognition",
                "affected_modules": "plate_recognition",
                "ai_generated": True,
                "fingerprint": "a1b2c3d4e5f6",
                "acknowledged": False,
            }
        }
