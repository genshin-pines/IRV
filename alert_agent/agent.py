"""
告警 Agent 决策引擎 — 日志监控主循环

职责:
  1. 定时从 LogCollector 拉取新日志
  2. 调用 LLM 进行日志分析 → 异常检测 → 级别判定 → 摘要生成
  3. 实现告警去重、升级/降级规则
  4. LLM 不可用时降级为纯规则引擎
  5. 输出告警事件，供 WebSocket/Webhook 推送

用法:
    from alert_agent import AlertAgent, create_agent

    agent = create_agent(client, collector)
    agent.start(interval=30)          # 每 30s 巡检一次
    # 或
    alerts = agent.run_once()         # 手动触发一次巡检
"""

import json
import time
import logging
import threading
import hashlib
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Callable

from .llm_client import LLMClient
from .logger import LogCollector, format_logs_for_llm
from .prompts import (
    SYSTEM_LOG_ANALYSIS_PROMPT,
    ANOMALY_DETECTION_PROMPT,
    ALERT_LEVEL_DECISION_PROMPT,
    ALERT_SUMMARY_PROMPT,
    COMBINED_ANALYSIS_PROMPT,
    COMBINED_DECISION_PROMPT,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 告警级别常量
# ═══════════════════════════════════════════════════════════════════

INFO = "info"
WARNING = "warning"
CRITICAL = "critical"

LEVEL_RANK = {INFO: 0, WARNING: 1, CRITICAL: 2}
LEVEL_ICON = {INFO: "🔵", WARNING: "🟡", CRITICAL: "🔴"}


# ═══════════════════════════════════════════════════════════════════
# 系统基线（正常阈值）
# ═══════════════════════════════════════════════════════════════════

SYSTEM_BASELINE = {
    "plate_confidence_min": 0.7,
    "api_latency_ms_max": 500,
    "inference_time_ms_max": 100,
    "camera_fps_min": 25,
    "error_rate_max": 0.01,
    "gesture_switch_max_per_10frame": 1,
}


# ═══════════════════════════════════════════════════════════════════
# 规则引擎（LLM 不可用时的 fallback）
# ═══════════════════════════════════════════════════════════════════

def rule_based_anomaly_detection(logs: List[Dict]) -> List[Dict]:
    """
    纯规则引擎检测异常（不依赖 LLM）。

    覆盖以下模式:
      - 连续低置信度车牌识别
      - 高频 ERROR/WARNING
      - API 超时
      - 摄像头帧率下降/断连
      - 手势频繁跳变
      - 连续登录失败
    """
    if not logs:
        return []

    anomalies = []
    error_count = sum(1 for e in logs if e["level"] in ("ERROR", "CRITICAL"))
    total = len(logs)

    # 1. 服务错误率突增
    if total > 10 and error_count / total > 0.1:
        anomalies.append({
            "id": _gen_alert_id("error_rate_high"),
            "type": "system",
            "severity_hint": "critical",
            "title": f"错误率异常: {error_count}/{total} ({error_count/total:.0%})",
            "description": f"最近 {total} 条日志中 ERROR 占比 {error_count/total:.0%}，超过 10% 阈值",
            "source_module": "system",
            "evidence": [e["message"] for e in logs if e["level"] in ("ERROR", "CRITICAL")][:5],
            "suggested_action": "检查各模块运行状态，优先排查高频错误来源",
            "baseline": f"< 1%",
            "current_value": f"{error_count/total:.0%}",
        })

    # 2. 车牌识别置信度偏低
    low_conf_plates = [
        e for e in logs
        if e["module"] == "plate_recognition"
        and "置信度:" in e["message"]
    ]
    if low_conf_plates:
        confs = []
        for e in low_conf_plates:
            try:
                conf_str = e["message"].split("置信度:")[1].split(",")[0].strip()
                confs.append(float(conf_str))
            except (ValueError, IndexError):
                pass
        low_confs = [c for c in confs if c < 0.5]
        if len(low_confs) >= 3:
            anomalies.append({
                "id": _gen_alert_id("plate_low_confidence"),
                "type": "recognition",
                "severity_hint": "warning",
                "title": f"车牌识别置信度持续偏低",
                "description": f"连续 {len(low_confs)} 帧置信度低于 0.5，平均 {sum(low_confs)/len(low_confs):.2f}",
                "source_module": "plate_recognition",
                "evidence": [e["message"] for e in low_conf_plates[:5]],
                "suggested_action": "检查摄像头画面是否模糊或光线不足",
                "baseline": f"> 0.7",
                "current_value": f"{sum(low_confs)/len(low_confs):.2f}",
            })

    # 3. 摄像头断连
    cam_disconnects = [
        e for e in logs
        if e["module"] == "camera_stream"
        and e["level"] in ("ERROR", "CRITICAL")
        and ("断连" in e["message"] or "中断" in e["message"] or "重连" in e["message"])
    ]
    if cam_disconnects:
        anomalies.append({
            "id": _gen_alert_id("camera_disconnect"),
            "type": "system",
            "severity_hint": "critical",
            "title": "摄像头流中断",
            "description": f"检测到 {len(cam_disconnects)} 次摄像头断连事件",
            "source_module": "camera_stream",
            "evidence": [e["message"] for e in cam_disconnects[:3]],
            "suggested_action": "检查 RTSP 服务器状态和网络连接",
            "baseline": "持续连接",
            "current_value": f"断连 {len(cam_disconnects)} 次",
        })

    # 4. API 超时
    timeouts = [
        e for e in logs
        if e["module"] == "api_server"
        and ("超时" in e["message"] or "timeout" in e["message"].lower())
    ]
    if timeouts:
        anomalies.append({
            "id": _gen_alert_id("api_timeout"),
            "type": "performance",
            "severity_hint": "warning" if len(timeouts) < 3 else "critical",
            "title": f"API 请求超时 ({len(timeouts)}次)",
            "description": f"最近 {len(timeouts)} 次 API 请求超时",
            "source_module": "api_server",
            "evidence": [e["message"] for e in timeouts[:3]],
            "suggested_action": "检查 API 服务负载和数据库连接",
            "baseline": f"< {SYSTEM_BASELINE['api_latency_ms_max']}ms",
            "current_value": "超时",
        })

    # 5. 手势跳变
    jitters = [
        e for e in logs
        if e["module"] == "gesture_recognition"
        and ("跳变" in e["message"] or "切换" in e["message"])
    ]
    if jitters:
        anomalies.append({
            "id": _gen_alert_id("gesture_jitter"),
            "type": "recognition",
            "severity_hint": "warning",
            "title": "手势识别频繁跳变",
            "description": f"检测到 {len(jitters)} 次手势识别结果跳变",
            "source_module": "gesture_recognition",
            "evidence": [e["message"] for e in jitters[:3]],
            "suggested_action": "检查关键点检测是否稳定，考虑增加平滑滤波",
            "baseline": "< 1次/10帧",
            "current_value": f"{len(jitters)} 次",
        })

    # 6. 连续登录失败
    login_fails = [
        e for e in logs
        if e["module"] == "auth"
        and ("失败" in e["message"] or "fail" in e["message"].lower())
    ]
    if len(login_fails) >= 3:
        anomalies.append({
            "id": _gen_alert_id("login_brute_force"),
            "type": "security",
            "severity_hint": "critical" if len(login_fails) >= 5 else "warning",
            "title": f"连续登录失败 ({len(login_fails)}次)",
            "description": f"检测到 {len(login_fails)} 次登录失败，可能存在暴力破解",
            "source_module": "auth",
            "evidence": [e["message"] for e in login_fails[:3]],
            "suggested_action": "检查登录来源 IP，考虑临时封禁或增加验证码",
            "baseline": "0",
            "current_value": f"{len(login_fails)} 次",
        })

    return anomalies


def rule_based_level_decision(anomalies: List[Dict], recent_alert_count: int = 0) -> Dict:
    """
    纯规则引擎判定告警级别（不依赖 LLM）。

    升级规则:
      - 任何 CRITICAL severity_hint → CRITICAL
      - 多个(≥2) WARNING → CRITICAL
      - 单个 WARNING → WARNING
      - 同一模块近期告警频繁 → 升级
    """
    if not anomalies:
        return {
            "decision": {"final_level": INFO, "is_upgraded": False},
            "alert": None,
        }

    severity_hints = [a.get("severity_hint", "info") for a in anomalies]
    has_critical = "critical" in severity_hints
    warning_count = severity_hints.count("warning")

    if has_critical:
        level = CRITICAL
        upgrade_reason = "存在严重级别异常"
    elif warning_count >= 2:
        level = CRITICAL
        upgrade_reason = f"多模块({warning_count}个)同时报告 WARNING，升级为严重"
    elif warning_count == 1:
        level = WARNING
        upgrade_reason = ""
    else:
        level = INFO
        upgrade_reason = ""

    # 近期告警频繁 → 升级
    if recent_alert_count >= 5 and level == WARNING:
        level = CRITICAL
        upgrade_reason = f"近1小时内已有 {recent_alert_count} 条告警，升级为严重"

    first = anomalies[0]
    return {
        "decision": {
            "final_level": level,
            "is_upgraded": level != first.get("severity_hint"),
            "upgrade_reason": upgrade_reason,
            "is_downgraded": False,
        },
        "alert": {
            "title": first.get("title", "系统异常"),
            "level": level,
            "icon": LEVEL_ICON.get(level, "🔵"),
            "summary": first.get("description", ""),
            "detail": "\n".join(a.get("description", "") for a in anomalies),
            "affected_modules": list(set(a.get("source_module", "") for a in anomalies)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ttl_minutes": 30 if level == CRITICAL else 60,
            "acknowledge_required": level == CRITICAL,
        },
        "context": {
            "recent_similar_alerts": recent_alert_count,
            "system_status": "需要关注" if level != INFO else "正常",
            "recommended_reviewer": "E",
        },
    }


# ═══════════════════════════════════════════════════════════════════
# Alert Agent 主类
# ═══════════════════════════════════════════════════════════════════

class AlertAgent:
    """
    告警智能体 — 持续监控日志并发布告警。

    工作模式:
      1. LLM 模式（默认）：日志 → LLM 分析 → 异常检测 → 级别判定 → 摘要生成
      2. 规则引擎模式（fallback）：LLM 不可用时自动切换，纯规则驱动

    Args:
        client: LLMClient 实例
        collector: LogCollector 实例
        use_llm: 是否使用 LLM（默认 True）
        dedup_window_minutes: 相同告警去重窗口（默认 30 分钟）
        max_logs_per_analysis: 单次分析最多处理多少条日志
        alert_callback: 告警回调，每产生一条告警时调用
                       签名: callback(alert_dict) -> None
    """

    def __init__(
        self,
        client: Optional[LLMClient],
        collector: LogCollector,
        *,
        use_llm: bool = True,
        dedup_window_minutes: int = 30,
        max_logs_per_analysis: int = 100,
        alert_callback: Optional[Callable[[Dict], None]] = None,
    ):
        self.client = client
        self.collector = collector
        self.use_llm = use_llm and client is not None
        self.dedup_window = timedelta(minutes=dedup_window_minutes)
        self.max_logs = max_logs_per_analysis
        self.alert_callback = alert_callback

        # 告警历史（内存），用于去重 + 升级判定
        self._alert_history: List[Dict] = []
        self._fingerprints: Dict[str, datetime] = {}
        self._lock = threading.Lock()

        # Agent 状态
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._llm_available = True
        self._cycle_count = 0
        self._error_count = 0
        self._started_at: Optional[datetime] = None

    # ── 公共 API ──────────────────────────────────────────────────

    def start(self, interval: int = 30):
        """
        启动后台定时巡检。

        Args:
            interval: 巡检间隔（秒），默认 30
        """
        if self._running:
            logger.warning("Agent 已在运行中")
            return

        self._running = True
        self._started_at = datetime.now(timezone.utc)
        self._thread = threading.Thread(
            target=self._loop,
            args=(interval,),
            daemon=True,
            name="AlertAgent",
        )
        self._thread.start()
        logger.info(f"Alert Agent 已启动, 巡检间隔={interval}s")

    def stop(self):
        """停止后台巡检"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Alert Agent 已停止")

    def run_once(self) -> List[Dict]:
        """
        手动触发一次巡检（同步）。

        Returns:
            本次产生的告警列表，每条包含完整决策信息
        """
        logs = self.collector.get_new_logs()
        if not logs:
            logger.debug("无新日志，跳过巡检")
            return []

        logger.info(f"巡检 #{self._cycle_count + 1}: 收到 {len(logs)} 条新日志")
        alerts = self._analyze(logs)

        for alert in alerts:
            self._emit_alert(alert)

        self._cycle_count += 1
        return alerts

    @property
    def status(self) -> Dict:
        """Agent 当前状态摘要"""
        with self._lock:
            recent = [
                a for a in self._alert_history
                if (datetime.now(timezone.utc) - a.get("_created_at", datetime.min.replace(tzinfo=timezone.utc))) < timedelta(hours=1)
            ]
        return {
            "running": self._running,
            "cycle_count": self._cycle_count,
            "error_count": self._error_count,
            "llm_available": self._llm_available,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "recent_alerts_1h": len(recent),
            "total_alerts": len(self._alert_history),
        }

    def recent_alerts(self, hours: int = 1) -> List[Dict]:
        """获取最近 N 小时内的告警"""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        with self._lock:
            return [
                a for a in self._alert_history
                if a.get("_created_at", datetime.min.replace(tzinfo=timezone.utc)) >= cutoff
            ]

    # ── 内部方法 ──────────────────────────────────────────────────

    def _loop(self, interval: int):
        """后台巡检主循环"""
        while self._running:
            try:
                self.run_once()
            except Exception as e:
                self._error_count += 1
                logger.error(f"Agent 巡检异常: {e}", exc_info=True)
            time.sleep(interval)

    def _analyze(self, logs: List[Dict]) -> List[Dict]:
        """
        分析日志，返回告警列表。

        优化策略（规则先行 + 合并 LLM 调用）:
          1. 规则引擎先行检测（0 延迟）
          2. 无异常 → 秒返，不调 LLM
          3. 有异常 → 合并 4 步为 2 步调用 LLM 深度分析
          4. LLM 不可用 → 规则引擎直接生产告警
        """
        # ── 快速通道：规则引擎先行 ──
        rule_anomalies = rule_based_anomaly_detection(logs)

        if not rule_anomalies:
            logger.debug(f"规则引擎未检测到异常，跳过 LLM 分析（快速通道）")
            self._llm_available = True  # 没有异常不等于 LLM 不可用
            return []

        logger.info(
            f"规则引擎检测到 {len(rule_anomalies)} 个异常，"
            f"将使用 LLM 深度分析..."
        )

        # ── LLM 通道：合并 4 步 → 2 步 ──
        if self.use_llm and self._llm_available and self.client:
            try:
                return self._analyze_with_llm_fast(logs, rule_anomalies)
            except Exception as e:
                logger.warning(f"LLM 分析失败，降级为规则引擎: {e}")
                self._llm_available = False
                self._error_count += 1

        return self._analyze_with_rules(logs)

    def _analyze_with_llm_fast(
        self, logs: List[Dict], rule_hints: List[Dict]
    ) -> List[Dict]:
        """
        合并 LLM 全链路分析（2 次调用替代 4 次）。

        Step A: COMBINED_ANALYSIS_PROMPT  → 日志解析 + 异常检测（合并）
        Step B: COMBINED_DECISION_PROMPT  → 级别判定 + 通知生成（合并）
        """
        if not self.client:
            return []

        formatted = format_logs_for_llm(logs, max_count=self.max_logs)

        # 附带规则引擎的预判结果，帮助 LLM 更快定位
        hints_text = json.dumps({
            "rule_engine_found": len(rule_hints),
            "hints": [
                {"type": h.get("type"), "title": h.get("title"),
                 "severity": h.get("severity_hint"), "module": h.get("source_module")}
                for h in rule_hints
            ],
        }, ensure_ascii=False)

        # ── Step A: 日志解析 + 异常检测（1 次调用） ──
        t0 = time.perf_counter()
        log_input = f"【规则引擎预判】\n{hints_text}\n\n【原始日志】\n{formatted}"
        combined_result = self.client.chat_json(
            user_message=log_input,
            system_prompt=COMBINED_ANALYSIS_PROMPT.replace("{log_text}", log_input),
        )
        logger.info(
            f"Step A (解析+检测) 耗时: {time.perf_counter() - t0:.1f}s"
        )

        anomalies = combined_result.get("anomalies", [])
        if not anomalies or not combined_result.get("is_anomalous"):
            # LLM 判定正常，但信任规则引擎发现的问题
            if rule_hints:
                logger.info(
                    f"LLM 判定正常，但规则引擎发现 {len(rule_hints)} 个异常（采纳规则引擎）"
                )
                return self._rule_decide_and_summarize(rule_hints)
            return []

        # 合并规则引擎 + LLM 发现的异常（去重）
        all_anomalies = _dedupe_anomalies(rule_hints + anomalies)

        # ── Step B: 级别判定 + 通知生成（1 次调用） ──
        t1 = time.perf_counter()
        anomaly_json = json.dumps({
            "anomalies": all_anomalies,
            "is_anomalous": True,
            "recent_alerts_1h": len(self.recent_alerts(hours=1)),
        }, ensure_ascii=False)

        try:
            decision = self.client.chat_json(
                user_message=anomaly_json,
                system_prompt=COMBINED_DECISION_PROMPT.replace(
                    "{anomaly_data}", anomaly_json
                ),
            )
            logger.info(
                f"Step B (判定+通知) 耗时: {time.perf_counter() - t1:.1f}s"
            )
        except Exception as e:
            logger.warning(f"LLM 决策失败，使用规则引擎: {e}")
            return self._rule_decide_and_summarize(all_anomalies)

        level = decision.get("final_level", WARNING)
        if level not in (INFO, WARNING, CRITICAL):
            level = WARNING

        if level == INFO:
            self._log_info_alert(decision.get("alert", {}))
            return []

        alert = {
            "level": level,
            "icon": LEVEL_ICON.get(level, "🔵"),
            "title": decision.get("alert", {}).get("title", "系统告警"),
            "summary": decision.get("alert", {}).get("summary", ""),
            "detail": decision.get("alert", {}).get("detail", ""),
            "affected_modules": decision.get("alert", {}).get("affected_modules", []),
            "decision": {"final_level": level},
            "anomalies": all_anomalies,
            "ai_generated": True,
            "acknowledge_required": (level == CRITICAL),
            "websocket": decision.get("websocket"),
            "webhook_markdown": decision.get("webhook_markdown"),
            "log_entry": decision.get("log_entry"),
        }
        return [alert]

    def _analyze_with_rules(self, logs: List[Dict]) -> List[Dict]:
        """规则引擎分析（LLM fallback）—— 每个异常生成一条告警"""
        anomalies = rule_based_anomaly_detection(logs)
        if not anomalies:
            return []

        recent_count = len(self.recent_alerts(hours=1))
        alerts = []

        for anomaly in anomalies:
            # 每个异常独立判定级别（考虑全局上下文）
            result = rule_based_level_decision(
                [anomaly],
                recent_alert_count=recent_count,
            )
            if result["alert"] is None:
                continue

            # 全局升级：多异常时提升级别
            level = result["alert"]["level"]
            if len(anomalies) >= 3 and level == WARNING:
                level = CRITICAL
            elif len(anomalies) >= 2:
                critical_hints = sum(
                    1 for a in anomalies
                    if a.get("severity_hint") == "critical"
                )
                if critical_hints >= 1:
                    level = CRITICAL

            alert = result["alert"]
            alert["level"] = level
            alert["icon"] = LEVEL_ICON.get(level, "🔵")
            alert["anomalies"] = [anomaly]   # 只关联自己的异常
            alert["ai_generated"] = False
            alert["acknowledge_required"] = (level == CRITICAL)

            summary = self._simple_summary({"alert": alert})
            alert.update(summary)
            alerts.append(alert)

        return alerts

    def _decide_and_summarize(self, anomalies: List[Dict]) -> List[Dict]:
        """
        LLM 后两步：级别判定 → 摘要生成。
        也支持纯规则引擎输出（当 self.use_llm=False 时）。
        """
        if not anomalies:
            return []

        if self.use_llm and self._llm_available and self.client:
            return self._llm_decide_and_summarize(anomalies)
        else:
            return self._rule_decide_and_summarize(anomalies)

    def _llm_decide_and_summarize(self, anomalies: List[Dict]) -> List[Dict]:
        """LLM 版本：级别判定 + 摘要生成"""
        if not self.client:
            return []

        # Step 3: 级别判定
        anomaly_json = json.dumps({
            "anomalies": anomalies,
            "is_anomalous": True,
            "recent_alerts_1h": len(self.recent_alerts(hours=1)),
        }, ensure_ascii=False)

        try:
            decision = self.client.chat_json(
                user_message=anomaly_json,
                system_prompt=ALERT_LEVEL_DECISION_PROMPT.replace(
                    "{anomaly_data}", anomaly_json
                ),
            )
        except Exception as e:
            logger.warning(f"LLM 级别判定失败，使用规则引擎: {e}")
            return self._rule_decide_and_summarize(anomalies)

        level = decision.get("decision", {}).get("final_level", WARNING)
        if level not in (INFO, WARNING, CRITICAL):
            level = WARNING

        # Step 4: 摘要生成（非 INFO 才生成通知）
        if level == INFO:
            alert = decision.get("alert", {})
            alert["anomalies"] = anomalies
            self._log_info_alert(alert)
            return []   # INFO 不推送

        try:
            summary = self.client.chat_json(
                user_message=json.dumps(decision, ensure_ascii=False),
                system_prompt=ALERT_SUMMARY_PROMPT.replace(
                    "{alert_decision}", json.dumps(decision, ensure_ascii=False)
                ),
            )
        except Exception as e:
            logger.warning(f"LLM 摘要生成失败，使用简单摘要: {e}")
            summary = self._simple_summary(decision)

        alert = {
            "level": level,
            "icon": LEVEL_ICON.get(level, "🔵"),
            "title": decision.get("alert", {}).get("title", "系统告警"),
            "summary": decision.get("alert", {}).get("summary", ""),
            "detail": decision.get("alert", {}).get("detail", ""),
            "affected_modules": decision.get("alert", {}).get("affected_modules", []),
            "decision": decision.get("decision", {}),
            "anomalies": anomalies,
            "ai_generated": True,
            "acknowledge_required": (level == CRITICAL),
            "websocket": summary.get("websocket"),
            "webhook_markdown": summary.get("webhook_markdown"),
            "log_entry": summary.get("log_entry"),
        }
        return [alert]

    def _rule_decide_and_summarize(self, anomalies: List[Dict]) -> List[Dict]:
        """规则引擎版本：级别判定 + 简单摘要"""
        result = rule_based_level_decision(
            anomalies,
            recent_alert_count=len(self.recent_alerts(hours=1)),
        )
        if result["alert"] is None:
            return []

        alert = result["alert"]
        alert["anomalies"] = anomalies
        alert["ai_generated"] = False

        if alert["level"] == INFO:
            self._log_info_alert(alert)
            return []

        # 生成简单通知文本
        summary = self._simple_summary(result)
        alert.update(summary)
        return [alert]

    def _emit_alert(self, alert: Dict):
        """发布一条告警：去重检查 → 记录 → 回调"""
        fingerprint = _alert_fingerprint(alert)

        with self._lock:
            # 去重
            last_time = self._fingerprints.get(fingerprint)
            now = datetime.now(timezone.utc)
            if last_time and (now - last_time) < self.dedup_window:
                logger.debug(f"告警去重: {alert.get('title')} (上次: {last_time.isoformat()})")
                return

            self._fingerprints[fingerprint] = now
            alert["_fingerprint"] = fingerprint
            alert["_created_at"] = now
            self._alert_history.append(alert)

            # 清理过期指纹
            expired = [
                fp for fp, t in self._fingerprints.items()
                if (now - t) > self.dedup_window
            ]
            for fp in expired:
                del self._fingerprints[fp]

        logger.info(
            f"[{LEVEL_ICON.get(alert['level'], '?')} {alert['level'].upper()}] "
            f"{alert.get('title', 'Unknown')}"
        )

        # 触发外部回调（WebSocket / Webhook）
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                logger.error(f"告警回调异常: {e}")

    @staticmethod
    def _log_info_alert(alert: Dict):
        """INFO 级别只记录不推送"""
        logger.info(
            f"[🔵 INFO] {alert.get('title', '')} — "
            f"{alert.get('summary', alert.get('detail', ''))[:100]}"
        )

    @staticmethod
    def _simple_summary(decision_or_result: Dict) -> Dict:
        """生成简单通知（LLM 不可用时的 fallback）"""
        alert = decision_or_result.get("alert", decision_or_result)
        level = alert.get("level", WARNING)
        title = alert.get("title", "系统异常")
        detail = alert.get("detail", alert.get("summary", ""))
        modules = alert.get("affected_modules", [])

        return {
            "websocket": {
                "type": "alert",
                "level": level,
                "title": title,
                "message": detail,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source_module": ", ".join(modules) if modules else "unknown",
                "suggested_action": alert.get("suggested_action", ""),
                "dismissible": True,
            },
            "webhook_markdown": (
                f"> {LEVEL_ICON.get(level, '')} **{title}**\n"
                f"> 时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"> 详情：{detail}\n"
                f"{'@所有人' if level == CRITICAL else ''}"
            ),
            "log_entry": (
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"[alert_agent] [{level.upper()}] {title} | {detail}"
            ),
        }


# ═══════════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════════

def _dedupe_anomalies(anomalies: List[Dict]) -> List[Dict]:
    """按 title + source_module 去重，保留首次出现的"""
    seen = set()
    result = []
    for a in anomalies:
        key = (a.get("title", ""), a.get("source_module", ""))
        if key not in seen:
            seen.add(key)
            result.append(a)
    return result


def _gen_alert_id(prefix: str) -> str:
    """生成告警唯一 ID"""
    ts = int(time.time() * 1000)
    return f"{prefix}_{ts}"


def _alert_fingerprint(alert: Dict) -> str:
    """生成告警指纹，用于去重"""

    key_parts = [
        alert.get("level", ""),
        alert.get("title", ""),
        ",".join(sorted(alert.get("affected_modules", []))),
    ]
    raw = "|".join(key_parts)
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def create_agent(
    client: Optional[LLMClient] = None,
    collector: Optional[LogCollector] = None,
    *,
    use_llm: bool = True,
    alert_callback: Optional[Callable[[Dict], None]] = None,
) -> AlertAgent:
    """
    创建 AlertAgent 的便捷工厂函数。

    Args:
        client: LLMClient 实例（为 None 时纯规则引擎模式）
        collector: LogCollector 实例（为 None 时自动创建）
        use_llm: 是否启用 LLM 分析
        alert_callback: 告警回调

    Returns:
        AlertAgent 实例

    Example:
        from alert_agent import create_agent, create_client

        client = create_client("deepseek")
        agent = create_agent(client, alert_callback=my_push_function)
        agent.start(interval=30)
    """
    if collector is None:
        collector = LogCollector(capacity=500)
        logging.getLogger().addHandler(collector)
        logging.getLogger().setLevel(logging.INFO)

    return AlertAgent(
        client=client,
        collector=collector,
        use_llm=use_llm,
        alert_callback=alert_callback,
    )
