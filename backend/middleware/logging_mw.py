from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from backend.services.log_service import write_log


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        level = "ERROR" if response.status_code >= 500 else "INFO"
        write_log("backend", level, f"{request.method} {request.url.path} {response.status_code} elapsed={elapsed_ms}ms")
        return response
