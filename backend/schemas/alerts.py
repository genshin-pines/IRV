from datetime import datetime
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.models.alert_event import AlertLevel, AlertStatus


T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    ok: bool = True
    data: T | None = None
    message: str = "success"
    trace_id: str


class AlertBase(BaseModel):
    level: AlertLevel
    title: str = Field(..., examples=["摄像头连接中断"])
    summary: str = ""
    source_module: str = Field(default="system", examples=["camera"])
    raw_log: str = ""
    llm_summary: str = ""
    ai_generated: bool = False
    status: AlertStatus = AlertStatus.UNREAD

    @field_validator("level", mode="before")
    @classmethod
    def normalize_level(cls, value):
        if isinstance(value, str):
            return value.upper()
        return value

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, value):
        if value is None or value == "":
            return AlertStatus.UNREAD
        if isinstance(value, str):
            return value.upper()
        return value

    @field_validator("summary", "source_module", "raw_log", "llm_summary", mode="before")
    @classmethod
    def none_to_empty(cls, value):
        return "" if value is None else value


class AlertCreate(AlertBase):
    pass


class AlertUpdate(BaseModel):
    status: AlertStatus | None = None
    title: str | None = None
    summary: str | None = None


class AlertRead(AlertBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime
    updated_at: datetime | None = None
    ack_time: datetime | None = None
    ack_user: str | None = None


class AlertAckRequest(BaseModel):
    ack_user: str = Field(default="admin", examples=["operator"])


class AlertList(BaseModel):
    items: list[AlertRead]
    total: int
    page: int
    page_size: int


class AlertStats(BaseModel):
    today_count: int
    total_count: int = 0
    unread_count: int
    error_count: int
    critical_count: int
    by_module: dict[str, int]
    by_level: dict[str, int] = Field(default_factory=dict)
    trend: list[dict[str, Any]]
    trend_4h: list[dict[str, Any]] = Field(default_factory=list)  # 兼容旧字段


class LogRead(BaseModel):
    seq: int
    timestamp: datetime
    module: str
    level: str
    message: str


class LogList(BaseModel):
    items: list[LogRead]
    total: int
    page: int
    page_size: int


class SimulateRequest(BaseModel):
    scenario: str = Field(default="mixed", examples=["plate_low_conf"])
    count: int = Field(default=10, ge=1, le=100)
