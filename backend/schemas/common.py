"""
公共 Pydantic Schema — 统一 API 响应格式
@owner 公共

所有接口统一返回:
  {"ok": true, "data": {...}, "message": "success", "trace_id": "..."}

用法:
    from backend.schemas.common import ok, fail

    @router.get("/api/plate/status")
    def plate_status():
        return ok(data={"running": True})

    @router.post("/api/plate/recognize-image")
    def recognize_image(file: UploadFile):
        if not file:
            return fail("未上传文件")
        return ok(data=result)
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class APIResponse(BaseModel):
    """统一 API 响应格式"""
    ok: bool = True
    data: Optional[Any] = None
    message: str = "success"
    trace_id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])


class PaginatedData(BaseModel):
    """分页数据"""
    items: list[Any] = []
    total: int = 0
    page: int = 1
    page_size: int = 20


# ── 快捷工厂 ──────────────────────────────────────────

def ok(data: Any = None, message: str = "success") -> dict:
    """成功响应"""
    return {
        "ok": True,
        "data": data,
        "message": message,
        "trace_id": _trace_id(),
    }


def fail(message: str, data: Any = None) -> dict:
    """失败响应"""
    return {
        "ok": False,
        "data": data,
        "message": message,
        "trace_id": _trace_id(),
    }


def _trace_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
