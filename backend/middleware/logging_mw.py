from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

from starlette.types import ASGIApp, Receive, Scope, Send

from backend.services.log_service import write_log


def _extract_username(scope: Scope) -> str:
    """从 Authorization Bearer 头提取用户名，失败返回 '-'"""
    try:
        for header_name, header_value in scope.get("headers", []):
            if header_name.decode("latin-1").lower() == "authorization":
                value = header_value.decode("latin-1")
                if value.startswith("Bearer "):
                    token = value[7:]
                    body_encoded, sig = token.rsplit(".", 1)
                    # 简单 HMAC 校验（与 preferences.py 一致的 demo token 格式）
                    expected_sig = hmac.new(
                        b"IRV-demo-secret-key",
                        body_encoded.encode("ascii"),
                        hashlib.sha256,
                    ).hexdigest()
                    if hmac.compare_digest(expected_sig, sig):
                        body = base64.urlsafe_b64decode(body_encoded + "==").decode("utf-8")
                        payload = json.loads(body)
                        return payload.get("sub", "-")
                    # HMAC 不匹配 — 携带了无效 token
                    write_log("auth", "WARNING", "token auth failed reason=signature_mismatch")
                break
    except Exception:
        write_log("auth", "WARNING", "token auth failed reason=parse_error")
    return "-"


class RequestLoggingMiddleware:
    """纯 ASGI 中间件 — 不使用 BaseHTTPMiddleware，避免破坏 StreamingResponse。

    BaseHTTPMiddleware 会用 anyio memory stream 包装响应体，导致 MJPEG 等
    无限 StreamingResponse 的 body 永远冲刷不到客户端（请求返回 200 但无数据）。
    详见: https://github.com/encode/starlette/issues/1192

    日志策略:
      - GET/HEAD/OPTIONS → module="backend", level=INFO（系统级请求日志）
      - POST/PUT/PATCH/DELETE → module="operation", level=INFO（用户操作日志）
      - 5xx → level=ERROR（异常日志，供告警规则消费）
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        start = time.perf_counter()
        status_code = 0
        path = scope.get("path", "")
        method = scope.get("method", "")

        async def send_wrapper(message: dict) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message.get("status", 0)
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            elapsed_ms = int((time.perf_counter() - start) * 1000)
            level = "ERROR" if status_code >= 500 else "INFO"

            # 写操作（POST/PUT/PATCH/DELETE）→ 用户操作日志
            if method in ("POST", "PUT", "PATCH", "DELETE"):
                username = _extract_username(scope)
                write_log("operation", level, f"user={username} {method} {path} {status_code} elapsed={elapsed_ms}ms")
            else:
                write_log("backend", level, f"{method} {path} {status_code} elapsed={elapsed_ms}ms")
