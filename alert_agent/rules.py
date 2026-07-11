from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Protocol

from alert_agent.templates import TEMPLATES
from backend.models.alert_event import AlertLevel
from backend.services.log_service import parse_latency_seconds


@dataclass(frozen=True)
class RuleResult:
    rule_id: str
    level: AlertLevel
    title: str
    summary: str
    source_module: str
    raw_logs: list[str]


class AlertRule(Protocol):
    rule_id: str

    def detect(self, logs: list[dict[str, Any]]) -> RuleResult | None:
        ...


def _message(log: dict[str, Any]) -> str:
    return str(log.get("message") or log.get("raw") or "")


class PlateLowConfidenceRule:
    rule_id = "plate_low_conf"

    def detect(self, logs: list[dict[str, Any]]) -> RuleResult | None:
        hits = []
        for log in logs:
            if log.get("module") != "plate":
                continue
            message = _message(log)
            match = re.search(r"(confidence|conf|置信度)\D*(0(?:\.\d+)?|1(?:\.0+)?)", message, re.I)
            if match and float(match.group(2)) < 0.75:
                hits.append(message)
        if len(hits) >= 5:
            return RuleResult(
                self.rule_id,
                AlertLevel.WARNING,
                "车牌识别置信度持续偏低",
                TEMPLATES["plate"],
                "plate",
                hits,
            )
        return None


class CameraDisconnectRule:
    rule_id = "camera_disconnect"
    keywords = ("Camera timeout", "RTSP disconnected", "Broken pipe", "Connection refused", "timeout", "断开", "中断")

    def detect(self, logs: list[dict[str, Any]]) -> RuleResult | None:
        hits = [
            _message(log)
            for log in logs
            if log.get("module") == "camera" and any(key.lower() in _message(log).lower() for key in self.keywords)
        ]
        if hits:
            return RuleResult(
                self.rule_id,
                AlertLevel.ERROR,
                "摄像头连接中断",
                TEMPLATES["camera"],
                "camera",
                hits,
            )
        return None


class GestureJitterRule:
    rule_id = "gesture_jitter"

    def detect(self, logs: list[dict[str, Any]]) -> RuleResult | None:
        gesture_logs = [log for log in logs if log.get("module") == "gesture"]
        recent = gesture_logs[-5:]
        values = []
        for log in recent:
            message = _message(log)
            match = re.search(r"(?:result|gesture|手势)\s*[:=：]\s*([\w\u4e00-\u9fff-]+)", message, re.I)
            if match:
                values.append(match.group(1))
            elif "跳变" in message or "jitter" in message.lower():
                values.append(message)
        changes = sum(1 for prev, curr in zip(values, values[1:]) if prev != curr)
        if len(values) >= 5 and changes > 3:
            return RuleResult(
                self.rule_id,
                AlertLevel.WARNING,
                "手势识别频繁跳变",
                TEMPLATES["gesture"],
                "gesture",
                [_message(log) for log in recent],
            )
        return None


class ApiTimeoutRule:
    rule_id = "api_timeout"

    def detect(self, logs: list[dict[str, Any]]) -> RuleResult | None:
        hits = []
        for log in logs:
            if log.get("module") not in {"backend", "llm"}:
                continue
            message = _message(log)
            lower = message.lower()
            if not any(name in lower for name in ("ocr", "gesture", "llm", "timeout", "超时")):
                continue
            elapsed = parse_latency_seconds(message)
            if elapsed is None and ("timeout" in lower or "超时" in message):
                hits.append(message)
            elif elapsed is not None and elapsed > 8:
                hits.append(message)
        if hits:
            return RuleResult(
                self.rule_id,
                AlertLevel.ERROR,
                "AI接口响应超时",
                TEMPLATES["timeout"],
                "backend",
                hits,
            )
        return None


class LoginFailRule:
    rule_id = "login_fail"

    def detect(self, logs: list[dict[str, Any]]) -> RuleResult | None:
        hits = [
            _message(log)
            for log in logs
            if log.get("module") == "login" and ("fail" in _message(log).lower() or "失败" in _message(log))
        ]
        if len(hits) >= 5:
            return RuleResult(
                self.rule_id,
                AlertLevel.WARNING,
                "连续登录失败",
                TEMPLATES["login"],
                "login",
                hits,
            )
        return None


class DriverAssistRiskRule:
    rule_id = "driver_assist_risk"

    def detect(self, logs: list[dict[str, Any]]) -> RuleResult | None:
        hits = []
        level = AlertLevel.WARNING
        for log in logs:
            message = _message(log)
            lower = message.lower()
            if "driver assist scene=" not in lower:
                continue
            if "near_collision" in lower:
                level = AlertLevel.CRITICAL
                hits.append(message)
            elif "traffic_police" in lower or "camera_disconnect" in lower:
                hits.append(message)
        if hits:
            title = "车外驾驶辅助风险提示" if level != AlertLevel.CRITICAL else "车外驾驶辅助高风险告警"
            return RuleResult(
                self.rule_id,
                level,
                title,
                "车外行车记录仪触发驾驶辅助风险，请结合车外画面、交警手势和车辆状态确认处置。",
                "camera",
                hits,
            )
        return None


class MixedAnomalyRule:
    rule_id = "mixed"

    def detect_from_results(self, results: list[RuleResult]) -> RuleResult | None:
        modules = {result.source_module for result in results}
        if len(results) >= 2 and len(modules) >= 2:
            raw_logs = [line for result in results for line in result.raw_logs[:2]]
            return RuleResult(
                self.rule_id,
                AlertLevel.CRITICAL,
                "系统存在复合异常",
                "多个模块同时发生异常，请优先排查摄像头、识别链路和后端依赖。",
                "system",
                raw_logs,
            )
        return None


class RuleEngine:
    def __init__(self) -> None:
        self.rules: list[AlertRule] = [
            PlateLowConfidenceRule(),
            CameraDisconnectRule(),
            GestureJitterRule(),
            ApiTimeoutRule(),
            LoginFailRule(),
            DriverAssistRiskRule(),
        ]
        self.mixed_rule = MixedAnomalyRule()

    def analyze(self, logs: list[dict[str, Any]]) -> list[RuleResult]:
        results = [result for rule in self.rules if (result := rule.detect(logs))]
        mixed = self.mixed_rule.detect_from_results(results)
        if mixed:
            results.append(mixed)
        return results
