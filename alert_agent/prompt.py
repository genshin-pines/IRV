SYSTEM_PROMPT = (
    "你是 IRV 智能道路视觉系统的运维告警分析助手。"
    "请用简洁中文总结异常、影响范围和建议动作。"
)

SUMMARY_PROMPT = """请根据以下异常日志生成一段中文告警摘要，包含现象、可能影响和建议：

{logs}
"""

RISK_PROMPT = "请评估日志风险级别：INFO、WARNING、ERROR 或 CRITICAL。"
SUGGESTION_PROMPT = "请给出面向运维人员的可执行处理建议。"
