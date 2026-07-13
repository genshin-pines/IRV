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
    _LOW_CONF_THRESHOLD = 0.98  # 低于此值视为低置信度

    def detect(self, logs: list[dict[str, Any]]) -> RuleResult | None:
        hits = []
        for log in logs:
            if log.get("module") != "plate":
                continue
            message = _message(log)
            # 匹配 "confidence=0.8500" / "conf: 0.72" / "置信度 0.80" 等格式
            match = re.search(r"(?:confidence|conf|置信度)\D*((?:0(?:\.\d+)?|1(?:\.0+)?))", message, re.I)
            if match:
                try:
                    conf_val = float(match.group(1))
                except ValueError:
                    continue
                if conf_val < self._LOW_CONF_THRESHOLD:
                    hits.append(message)
        if len(hits) >= 3:
            return RuleResult(
                self.rule_id,
                AlertLevel.WARNING,
                f"车牌识别置信度持续偏低（{len(hits)} 次低置信度命中）",
                TEMPLATES["plate"],
                "plate",
                hits[:10],  # 只取前 10 条稳定指纹，避免窗口内累积导致反复触发
            )
        return None


class CameraDisconnectRule:
    rule_id = "camera_disconnect"
    keywords = ("Camera timeout", "RTSP disconnected", "Broken pipe", "Connection refused", "断开", "中断")

    def detect(self, logs: list[dict[str, Any]]) -> RuleResult | None:
        hits = [
            _message(log)
            for log in logs
            if log.get("module") == "camera"
            and log.get("level") in ("ERROR", "CRITICAL")
            and any(key.lower() in _message(log).lower() for key in self.keywords)
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
            match = re.search(r"(?:result|gesture|手势|type)\s*[:=：]\s*([\w\u4e00-\u9fff-]+)", message, re.I)
            if match:
                values.append(match.group(1))
            elif "跳变" in message or "jitter" in message.lower():
                values.append(message)
        changes = sum(1 for prev, curr in zip(values, values[1:]) if prev != curr)
        if len(values) >= 2 and changes >= 1:  # TODO-TEST: revert to len>=5 and changes>3
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
            elif elapsed is not None and elapsed > 3:
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


class PlatePipelineFailureRule:
    """车牌识别管线硬失败 — 图片解码失败、视频无法打开、模型加载失败等"""
    rule_id = "plate_pipeline_failure"

    def detect(self, logs: list[dict[str, Any]]) -> RuleResult | None:
        hits = [
            _message(log)
            for log in logs
            if log.get("module") in {"plate", "camera"}
            and log.get("level") == "ERROR"
            and any(kw in _message(log).lower() for kw in (
                "decode failed", "cannot open", "recognition failed",
                "加载失败", "warmup failed",
            ))
        ]
        if hits:
            return RuleResult(
                self.rule_id,
                AlertLevel.ERROR,
                "车牌识别管线异常",
                TEMPLATES["plate_pipeline"],
                "plate",
                hits,
            )
        return None


# ═══════════════════════════════════════════════════════════
# P0 — 审计新增规则
# ═══════════════════════════════════════════════════════════

class LLMDegradationRule:
    """LLM 降级 / 鉴权失败 / Token 超额告警"""
    rule_id = "llm_degradation"

    _TOKEN_KW = ("token", "context length", "maximum context", "too long",
                 "reduce the length", "exceed", "上下文", "超限")

    def detect(self, logs: list[dict[str, Any]]) -> RuleResult | None:
        hits: list[str] = []
        has_critical = False
        token_limit_hit = False
        for log in logs:
            if log.get("module") not in {"llm", "fusion"}:
                continue
            message = _message(log)
            lower = message.lower()
            if any(kw in lower for kw in ("llm downgraded", "llm 连续失败", "降级为纯规则模式")):
                hits.append(message)
                has_critical = True
                if any(kw in lower for kw in self._TOKEN_KW):
                    token_limit_hit = True
            elif "llm 融合推理失败" in lower:
                hits.append(message)
            elif log.get("level") in ("ERROR", "CRITICAL") and any(
                kw in lower for kw in ("401", "403", "429", "api key", "unauthorized", "rate limit")
            ):
                hits.append(message)
                has_critical = True
        if hits:
            if token_limit_hit:
                title = "LLM Token 超额或上下文超限"
                summary = TEMPLATES["llm_token"]
            else:
                title = "LLM 接口异常或已降级"
                summary = TEMPLATES["llm"]
            return RuleResult(
                self.rule_id,
                AlertLevel.ERROR if has_critical else AlertLevel.WARNING,
                title,
                summary,
                "llm",
                hits,
            )
        return None


class FusionExceptionRule:
    """融合引擎异常 — fusion_agent ERROR 日志，无规则监听"""
    rule_id = "fusion_exception"

    def detect(self, logs: list[dict[str, Any]]) -> RuleResult | None:
        hits = [
            _message(log)
            for log in logs
            if log.get("module") == "fusion"
            and log.get("level") in ("ERROR", "CRITICAL")
            and any(kw in _message(log).lower() for kw in ("融合推理异常", "结果回调异常", "fusion"))
        ]
        if hits:
            return RuleResult(
                self.rule_id,
                AlertLevel.ERROR,
                "融合推理引擎异常",
                TEMPLATES["fusion"],
                "fusion",
                hits,
            )
        return None


# ═══════════════════════════════════════════════════════════
# P1 — 审计新增规则
# ═══════════════════════════════════════════════════════════

class GestureLowConfidenceRule:
    """手势置信度持续低 / 手势管线硬失败"""
    rule_id = "gesture_low_conf"

    _CONFIDENCE_KW = ("confidence low", "置信度低", "low confidence")

    def detect(self, logs: list[dict[str, Any]]) -> RuleResult | None:
        # 管线硬失败 — 1 条即触发 ERROR
        error_hits = [
            _message(log)
            for log in logs
            if log.get("module") == "gesture"
            and log.get("level") == "ERROR"
            and any(kw in _message(log).lower() for kw in ("decode failed",))
        ]
        if error_hits:
            return RuleResult(
                self.rule_id,
                AlertLevel.ERROR,
                "手势识别管线异常",
                TEMPLATES["gesture"],
                "gesture",
                error_hits,
            )
        # 置信度持续低 — ≥3 条触发 WARNING
        warn_hits = [
            _message(log)
            for log in logs
            if log.get("module") == "gesture"
            and log.get("level") in ("WARNING", "ERROR")
            and any(kw in _message(log).lower() for kw in self._CONFIDENCE_KW)
        ]
        if len(warn_hits) >= 2:
            return RuleResult(
                self.rule_id,
                AlertLevel.WARNING,
                f"手势识别置信度持续偏低（{len(warn_hits)} 次低置信度命中）",
                TEMPLATES["gesture"],
                "gesture",
                warn_hits,
            )
        return None


class GestureFalseTriggerRule:
    """手势误触发率监控 — stable=False 占比过高"""
    rule_id = "gesture_false_trigger"
    _WINDOW = 10
    _THRESHOLD = 0.3

    def detect(self, logs: list[dict[str, Any]]) -> RuleResult | None:
        gesture_logs = [log for log in logs if log.get("module") == "gesture"]
        if len(gesture_logs) < self._WINDOW:
            return None
        recent = gesture_logs[-self._WINDOW:]
        unstable = sum(1 for log in recent if "stable=false" in _message(log).lower())
        ratio = unstable / self._WINDOW
        if ratio >= self._THRESHOLD:
            return RuleResult(
                self.rule_id,
                AlertLevel.WARNING,
                f"手势误触发率偏高（不稳定占比 {ratio:.0%}，最近 {self._WINDOW} 次中 {unstable} 次不稳定）",
                TEMPLATES["gesture_false_trigger"],
                "gesture",
                [_message(log) for log in recent if "stable=false" in _message(log).lower()],
            )
        return None


class DatabaseExceptionRule:
    """数据库异常 — 模板已有、无规则监听"""
    rule_id = "database_exception"

    def detect(self, logs: list[dict[str, Any]]) -> RuleResult | None:
        hits = [
            _message(log)
            for log in logs
            if log.get("module") in {"system", "backend"}
            and log.get("level") in ("ERROR", "CRITICAL")
            and any(kw in _message(log).lower() for kw in (
                "sql", "database", "sqlite", "operationalerror", "integrityerror",
                "磁盘满", "连接池", "db error", "disk full",
            ))
        ]
        if hits:
            return RuleResult(
                self.rule_id,
                AlertLevel.ERROR,
                "数据库操作异常",
                TEMPLATES["database"],
                "system",
                hits,
            )
        return None


class TrafficPoliceAnomalyRule:
    """交警手势识别异常 — 模型加载失败、流启动失败、置信度持续低"""
    rule_id = "traffic_police_anomaly"

    def detect(self, logs: list[dict[str, Any]]) -> RuleResult | None:
        error_hits = [
            _message(log)
            for log in logs
            if log.get("module") == "traffic_police"
            and log.get("level") in ("ERROR", "CRITICAL")
        ]
        warn_hits = [
            _message(log)
            for log in logs
            if log.get("module") == "traffic_police"
            and log.get("level") == "WARNING"
        ]
        if error_hits:
            return RuleResult(
                self.rule_id,
                AlertLevel.ERROR,
                "交警手势识别异常",
                TEMPLATES["traffic_police"],
                "traffic_police",
                error_hits,
            )
        if len(warn_hits) >= 2:
            return RuleResult(
                self.rule_id,
                AlertLevel.WARNING,
                f"交警手势识别置信度持续偏低（{len(warn_hits)} 次）",
                TEMPLATES["traffic_police"],
                "traffic_police",
                warn_hits,
            )
        return None


class UnauthorizedAccessRule:
    """未授权 API 访问 — token 校验失败"""
    rule_id = "unauthorized_access"

    def detect(self, logs: list[dict[str, Any]]) -> RuleResult | None:
        hits = [
            _message(log)
            for log in logs
            if log.get("module") == "auth"
            and log.get("level") == "WARNING"
            and "token auth failed" in _message(log).lower()
        ]
        if len(hits) >= 1:
            return RuleResult(
                self.rule_id,
                AlertLevel.WARNING,
                f"存在未授权 API 访问尝试（{len(hits)} 次 token 校验失败）",
                TEMPLATES["unauthorized"],
                "auth",
                hits,
            )
        return None


class NetworkExceptionRule:
    """网络通用异常 — 模板已有、CameraDisconnectRule 未覆盖的通用网络故障"""
    rule_id = "network_exception"

    def detect(self, logs: list[dict[str, Any]]) -> RuleResult | None:
        hits = [
            _message(log)
            for log in logs
            # 排除 camera 模块（已由 CameraDisconnectRule 覆盖）
            # 排除 backend/llm 的 timeout（已由 ApiTimeoutRule 覆盖）
            if log.get("module") not in {"camera", "backend", "llm"}
            and log.get("level") in ("ERROR", "CRITICAL")
            and any(kw in _message(log).lower() for kw in (
                "connection refused", "connection reset", "connectionerror",
                "dns", "unreachable", "network", "网络", "econnrefused",
                "econnreset", "name or service not known",
            ))
        ]
        if hits:
            return RuleResult(
                self.rule_id,
                AlertLevel.ERROR,
                "网络连接异常",
                TEMPLATES["network"],
                "system",
                hits,
            )
        return None


class RuleEngine:
    def __init__(self) -> None:
        self.rules: list[AlertRule] = [
            PlateLowConfidenceRule(),
            PlatePipelineFailureRule(),
            CameraDisconnectRule(),
            GestureJitterRule(),
            ApiTimeoutRule(),
            LoginFailRule(),
            DriverAssistRiskRule(),
            # P0 新增
            LLMDegradationRule(),
            FusionExceptionRule(),
            # P1 新增
            GestureLowConfidenceRule(),
            GestureFalseTriggerRule(),
            DatabaseExceptionRule(),
            UnauthorizedAccessRule(),
            TrafficPoliceAnomalyRule(),
            NetworkExceptionRule(),
        ]
        self.mixed_rule = MixedAnomalyRule()

    def analyze(self, logs: list[dict[str, Any]]) -> list[RuleResult]:
        results = [result for rule in self.rules if (result := rule.detect(logs))]
        mixed = self.mixed_rule.detect_from_results(results)
        if mixed:
            results.append(mixed)
        return results
