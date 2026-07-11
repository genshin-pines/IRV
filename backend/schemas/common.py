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

import json
import math
import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field
from starlette.responses import JSONResponse


# ── NaN/Infinity 安全 JSON 响应 ───────────────────────

def _sanitize_float(obj: Any) -> Any:
    """递归替换 NaN / Infinity 为 None，确保 JSON 可序列化"""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_float(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_float(v) for v in obj]
    return obj


class SanitizedJSONResponse(JSONResponse):
    """自动清理 NaN / Infinity 的 JSONResponse"""

    def render(self, content: Any) -> bytes:
        return json.dumps(
            _sanitize_float(content),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")


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
