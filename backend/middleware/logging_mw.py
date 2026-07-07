"""
请求日志中间件 — 自动捕获每个 HTTP 请求并输出到 Python logging
@owner 成员E

通过 Python logging 机制，这些日志会被 LogCollector 自动收集。
"""

import time
import logging

logger = logging.getLogger("api_server")


class RequestLoggingMiddleware:
    """ASGI 中间件：记录每个 HTTP 请求的方法、路径、状态码、耗时"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        t0 = time.perf_counter()
        method = scope.get("method", "?")
        path = scope.get("path", "/")

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status = message["status"]
                elapsed_ms = round((time.perf_counter() - t0) * 1000)

                if status >= 500:
                    logger.error(f"{method} {path} → {status} {elapsed_ms}ms")
                elif status >= 400:
                    logger.warning(f"{method} {path} → {status} {elapsed_ms}ms")
                elif elapsed_ms > 2000:
                    logger.warning(f"请求超时: {method} {path}, 耗时: {elapsed_ms}ms")
                else:
                    logger.info(f"{method} {path} → {status} {elapsed_ms}ms")

            await send(message)

        await self.app(scope, receive, send_wrapper)
