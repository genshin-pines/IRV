"""
告警 Agent 模块 — 日志监控与告警智能体

职责:
  1. 监听系统日志（车牌识别、手势识别、API 请求等）
  2. 调用 LLM 分析日志，检测异常模式
  3. 自主决策告警级别（提示/警告/严重）
  4. 生成自然语言告警摘要
  5. 通过 WebSocket / Webhook 推送告警通知

支持的 LLM Provider:
  - DeepSeek-V4
  - Kimi (Moonshot)
  - GPT-4o (OpenAI)

使用示例:
    from alert_agent import create_client, create_agent, LogCollector

    # 1. 创建 LLM 客户端
    client = create_client("deepseek")

    # 2. 创建日志收集器
    collector = LogCollector(capacity=500)
    import logging
    logging.getLogger().addHandler(collector)

    # 3. 创建 Agent 并启动
    agent = create_agent(client, collector, alert_callback=my_callback)
    agent.start(interval=30)
"""

from .config import LLMConfig, get_config, PRESET_PROVIDERS
from .llm_client import LLMClient, create_client
from .logger import LogCollector, simulate_logs, format_logs_for_llm
from .agent import (
    AlertAgent,
    create_agent,
    rule_based_anomaly_detection,
    rule_based_level_decision,
    INFO,
    WARNING,
    CRITICAL,
    LEVEL_ICON,
)
from .prompts import (
    SYSTEM_LOG_ANALYSIS_PROMPT,
    ANOMALY_DETECTION_PROMPT,
    ALERT_LEVEL_DECISION_PROMPT,
    ALERT_SUMMARY_PROMPT,
    REACT_AGENT_SYSTEM_PROMPT,
    PLATE_RECOGNITION_ANALYSIS_PROMPT,
    GESTURE_RECOGNITION_ANALYSIS_PROMPT,
    API_SERVICE_ANALYSIS_PROMPT,
    COMBINED_ANALYSIS_PROMPT,
    COMBINED_DECISION_PROMPT,
)

__all__ = [
    # ── 配置 ──
    "LLMConfig",
    "get_config",
    "PRESET_PROVIDERS",
    # ── LLM 客户端 ──
    "LLMClient",
    "create_client",
    # ── 日志收集 ──
    "LogCollector",
    "simulate_logs",
    "format_logs_for_llm",
    # ── Agent 引擎 ──
    "AlertAgent",
    "create_agent",
    "rule_based_anomaly_detection",
    "rule_based_level_decision",
    "INFO",
    "WARNING",
    "CRITICAL",
    "LEVEL_ICON",
    # ── Prompt 模板 ──
    "SYSTEM_LOG_ANALYSIS_PROMPT",
    "ANOMALY_DETECTION_PROMPT",
    "ALERT_LEVEL_DECISION_PROMPT",
    "ALERT_SUMMARY_PROMPT",
    "REACT_AGENT_SYSTEM_PROMPT",
    "PLATE_RECOGNITION_ANALYSIS_PROMPT",
    "GESTURE_RECOGNITION_ANALYSIS_PROMPT",
    "API_SERVICE_ANALYSIS_PROMPT",
    "COMBINED_ANALYSIS_PROMPT",
    "COMBINED_DECISION_PROMPT",
]
