from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import logging
from logging.handlers import RotatingFileHandler
import re
import threading
from typing import Any, Callable

from backend.config import LOG_FILE, LOG_LEVEL, LOG_COLLECTOR_CAPACITY


LOG_SOURCES = {"plate", "gesture", "camera", "backend", "auth", "login", "llm", "traffic_police", "fusion", "operation", "system", "mobile"}
logger = logging.getLogger("alert_agent")


class MemoryLogHandler(logging.Handler):
    def __init__(self, capacity: int = LOG_COLLECTOR_CAPACITY) -> None:
        super().__init__()
        self._items: deque[dict[str, Any]] = deque(maxlen=capacity)
        self._seq = 0
        self._last_read_seq = 0
        self._lock = threading.Lock()
        self.on_emit: Callable[[], None] | None = None

    def emit(self, record: logging.LogRecord) -> None:
        with self._lock:
            self._seq += 1
            entry = {
                "seq": self._seq,
                "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc),
                "module": normalize_module(record.name),
                "level": record.levelname,
                "message": record.getMessage(),
                "raw": self.format(record),
            }
            self._items.append(entry)
        callback = self.on_emit
        if callback is not None:
            try:
                callback()
            except Exception:
                pass

    def query(
        self,
        *,
        page: int = 1,
        page_size: int = 50,
        module: str | None = None,
        level: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        with self._lock:
            items = list(self._items)
        if module:
            items = [item for item in items if item["module"] == normalize_module(module)]
        if level:
            items = [item for item in items if item["level"] == level.upper()]
        if start_time:
            items = [item for item in items if item["timestamp"] >= start_time]
        if end_time:
            items = [item for item in items if item["timestamp"] <= end_time]
        total = len(items)
        start = (page - 1) * page_size
        return items[start:start + page_size], total

    def get_new_logs(self) -> list[dict[str, Any]]:
        with self._lock:
            items = [item for item in self._items if item["seq"] > self._last_read_seq]
            if items:
                self._last_read_seq = items[-1]["seq"]
            return items

    def stats(self) -> dict[str, Any]:
        with self._lock:
            items = list(self._items)
        by_level: dict[str, int] = {}
        by_module: dict[str, int] = {}
        for item in items:
            by_level[item["level"]] = by_level.get(item["level"], 0) + 1
            by_module[item["module"]] = by_module.get(item["module"], 0) + 1
        return {"total": len(items), "by_level": by_level, "by_module": by_module}


_collector: MemoryLogHandler | None = None


def setup_log_collector() -> MemoryLogHandler:
    global _collector
    if _collector is not None:
        return _collector

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _collector = MemoryLogHandler()
    _collector.setFormatter(formatter)

    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    if not any(isinstance(handler, MemoryLogHandler) for handler in root.handlers):
        root.addHandler(_collector)
    if not any(isinstance(handler, RotatingFileHandler) and handler.baseFilename == str(LOG_FILE) for handler in root.handlers):
        root.addHandler(file_handler)

    logger.info("alert log collector initialized")
    return _collector


def get_collector() -> MemoryLogHandler:
    return setup_log_collector()


def query_logs(**kwargs: Any) -> tuple[list[dict[str, Any]], int]:
    return get_collector().query(**kwargs)


def get_log_stats() -> dict[str, Any]:
    return get_collector().stats()


def write_log(source: str, level: str, message: str) -> None:
    logging.getLogger(normalize_module(source)).log(getattr(logging, level.upper()), message)


def normalize_module(name: str) -> str:
    lowered = name.lower()
    aliases = {
        "plate": "plate",
        "plate_recognition": "plate",
        "gesture": "gesture",
        "gesture_recognition": "gesture",
        "camera": "camera",
        "camera_stream": "camera",
        "auth": "auth",
        "login": "login",
        "api": "backend",
        "backend": "backend",
        "llm": "llm",
        "fusion": "fusion",
        "fusion_agent": "fusion",
        "traffic_police": "traffic_police",
        "operation": "operation",
        "mobile": "mobile",
    }
    for key, value in aliases.items():
        if key in lowered:
            return value
    return "system"


def parse_latency_seconds(message: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(ms|毫秒|s|sec|秒)", message, re.I)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    return value / 1000 if unit in {"ms", "毫秒"} else value
