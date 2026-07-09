"""
告警 Agent 演示脚本 — Day 1 验证

演示:
  1. LLM 客户端初始化（支持 DeepSeek / Kimi / OpenAI）
  2. 模拟日志生成 → 格式化 → 送入 Prompt 模板
  3. 调用 LLM 进行日志分析、异常检测、告警判定
  4. 生成最终告警通知（WebSocket / Webhook / 日志文件）

用法:
  # 仅验证配置和 prompt 模板（不调用 LLM）
  python -m alert_agent.demo --dry-run

  # 实际调用 DeepSeek API
  $env:DEEPSEEK_API_KEY = "sk-xxx"
  python -m alert_agent.demo

  # 使用其他 provider
  $env:OPENAI_API_KEY = "sk-xxx"
  python -m alert_agent.demo --provider openai
"""

import sys
import json
import logging
from pathlib import Path

# 修复 Windows GBK 终端编码问题
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alert_agent.config import get_config, PRESET_PROVIDERS
from alert_agent.llm_client import create_client
from alert_agent.logger import LogCollector, simulate_logs, format_logs_for_llm
from alert_agent.prompts import (
    SYSTEM_LOG_ANALYSIS_PROMPT,
    ANOMALY_DETECTION_PROMPT,
    ALERT_LEVEL_DECISION_PROMPT,
    ALERT_SUMMARY_PROMPT,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("demo")


def step_header(title: str):
    """打印步骤标题"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def demo_dry_run():
    """
    干跑模式：
    - 不实际调用 LLM API
    - 展示完整的 Prompt 模板填充效果
    - 验证整个流程可以跑通
    """
    step_header("Step 1: 检查 LLM Provider 配置")
    for name, cfg in PRESET_PROVIDERS.items():
        has_key = bool(cfg.api_key)
        status = "[OK] 已配置" if has_key else "[!!] 未设置 (需配置环境变量)"
        print(f"  {name:12s} → {cfg.base_url}  [{cfg.model}]  {status}")

    step_header("Step 2: 模拟日志生成")
    collector = LogCollector(capacity=200)
    simulate_logs(collector, count=15)
    logs = collector.get_all_logs()
    print(f"  生成了 {len(logs)} 条模拟日志")

    # 展示部分日志
    print(f"\n  最近 8 条日志:")
    for entry in logs[-8:]:
        level_icon = {"INFO": "  ", "WARNING": "!!", "ERROR": "XX"}.get(entry["level"], "  ")
        print(f"    {level_icon} [{entry['module']:20s}] [{entry['level']:7s}] {entry['message'][:60]}")

    stats = collector.get_stats()
    print(f"\n  日志统计: {json.dumps(stats, ensure_ascii=False, indent=2)}")

    step_header("Step 3: Prompt 模板填充演示")
    formatted_logs = format_logs_for_llm(logs)

    # 3.1 日志解析 Prompt
    print("  [3.1] 系统日志解析 Prompt")
    filled = SYSTEM_LOG_ANALYSIS_PROMPT.replace("{log_text}", formatted_logs)
    print(f"    填充后长度: {len(filled)} chars")
    print(f"    预览 (前200字): {filled[:200]}...")

    # 3.2 异常检测 Prompt（用统计信息模拟结构化数据）
    print("\n  [3.2] 异常检测 Prompt")
    log_data_json = json.dumps({
        "logs": logs,
        "stats": stats,
    }, ensure_ascii=False, indent=2)
    filled = ANOMALY_DETECTION_PROMPT.replace("{log_data}", log_data_json[:1500] + "\n...")
    print(f"    填充后长度: {len(filled)} chars")

    # 3.3 告警级别判定 Prompt（模拟异常输入）
    print("\n  [3.3] 告警级别判定 Prompt")
    mock_anomalies = json.dumps({
        "anomalies": [
            {
                "type": "recognition",
                "severity_hint": "warning",
                "title": "车牌识别置信度持续偏低",
                "description": "过去5分钟内有3帧置信度低于0.5，当前平均0.40",
                "source_module": "plate_recognition",
                "evidence": [
                    "[14:30:15] 川D54321 置信度: 0.42",
                    "[14:30:45] 浙F88888 置信度: 0.38",
                ],
                "suggested_action": "检查摄像头画面质量",
                "baseline": "≥0.7",
                "current_value": "0.40",
            },
            {
                "type": "performance",
                "severity_hint": "warning",
                "title": "API 请求超时",
                "description": "POST /recognize-video 耗时 3200ms",
                "source_module": "api_server",
            },
        ],
        "is_anomalous": True,
        "overall_trend": "需要关注",
        "analysis_confidence": 0.85,
    }, ensure_ascii=False, indent=2)
    filled = ALERT_LEVEL_DECISION_PROMPT.replace("{anomaly_data}", mock_anomalies)
    print(f"    填充后长度: {len(filled)} chars")
    print(f"    决策输入: {mock_anomalies[:300]}...")

    # 3.4 告警摘要 Prompt
    print("\n  [3.4] 告警摘要 Prompt")
    mock_decision = json.dumps({
        "decision": {
            "final_level": "warning",
            "is_upgraded": False,
            "is_downgraded": False,
        },
        "alert": {
            "title": "车牌识别置信度异常",
            "level": "warning",
            "icon": "🟡",
            "summary": "车牌识别置信度连续3帧低于0.5",
            "detail": "过去5分钟内，plate_recognition模块出现连续低置信度识别，当前平均0.40，远低于正常基线0.7",
            "affected_modules": ["plate_recognition"],
            "acknowledge_required": False,
        },
    }, ensure_ascii=False, indent=2)
    filled = ALERT_SUMMARY_PROMPT.replace("{alert_decision}", mock_decision)
    print(f"    填充后长度: {len(filled)} chars")

    step_header("Step 4: 完整流程总结")
    print("""
  ┌─────────────────────────────────────────────────┐
  │  告警 Agent 处理流水线 (Day 1 验证完成)          │
  │                                                  │
  │  1. 日志源 ──→ LogCollector 捕获                 │
  │  2. 日志文本 ──→ SYSTEM_LOG_ANALYSIS_PROMPT      │
  │                 → LLM 解析为结构化 JSON           │
  │  3. 结构化数据 ──→ ANOMALY_DETECTION_PROMPT       │
  │                   → LLM 检测异常模式              │
  │  4. 异常列表 ──→ ALERT_LEVEL_DECISION_PROMPT      │
  │                 → LLM 判定告警级别                │
  │  5. 告警决策 ──→ ALERT_SUMMARY_PROMPT             │
  │                 → LLM 生成多格式通知              │
  │  6. 通知 ──→ WebSocket / Webhook / 日志文件       │
  │                                                  │
  │  支持 LLM: DeepSeek-V4 / Kimi / GPT-4o           │
  │  配置方式: 环境变量 {PROVIDER}_API_KEY            │
  │                                                  │
  │  [OK] Day 1 任务: 调研 LLM API + Prompt 模板 完成   │
  └─────────────────────────────────────────────────┘
""")


def demo_live(provider: str = "deepseek"):
    """实际调用 LLM API 的完整演示"""
    step_header(f"Step 1: 连接 LLM ({provider})")

    try:
        client = create_client(provider)
        print(f"  [OK] 客户端已创建: model={client.config.model}")
    except ValueError as e:
        print(f"  [XX] 配置错误: {e}")
        print(f"  提示: 设置环境变量后重试")
        print(f"    PowerShell: $env:{provider.upper()}_API_KEY = 'your-key'")
        return

    step_header("Step 2: 生成模拟日志并分析")
    collector = LogCollector(capacity=200)
    simulate_logs(collector, count=12)
    logs = collector.get_all_logs()
    formatted = format_logs_for_llm(logs)

    print(f"  已生成 {len(logs)} 条模拟日志")
    print()

    # 调用 LLM 做日志解析
    print("  发送日志分析请求...")
    try:
        result = client.chat_json(
            user_message=formatted,
            system_prompt=SYSTEM_LOG_ANALYSIS_PROMPT.replace("{log_text}", formatted),
        )
        print(f"  [OK] 日志分析结果:")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"  [XX] 日志分析失败: {e}")
        return

    step_header("Step 3: 异常检测")
    try:
        anomalies = client.chat_json(
            user_message=json.dumps(result, ensure_ascii=False),
            system_prompt=ANOMALY_DETECTION_PROMPT.replace(
                "{log_data}", json.dumps(result, ensure_ascii=False)
            ),
        )
        print(f"  是否异常: {anomalies.get('is_anomalous')}")
        print(f"  检测到 {len(anomalies.get('anomalies', []))} 个异常")
        if anomalies.get("anomalies"):
            for a in anomalies["anomalies"]:
                print(f"    - [{a.get('severity_hint')}] {a.get('title')}")
    except Exception as e:
        print(f"  [XX] 异常检测失败: {e}")
        return

    if not anomalies.get("is_anomalous"):
        print("  [OK] 系统状态正常，无需告警")
        return

    step_header("Step 4: 告警级别判定")
    try:
        decision = client.chat_json(
            user_message=json.dumps(anomalies, ensure_ascii=False),
            system_prompt=ALERT_LEVEL_DECISION_PROMPT.replace(
                "{anomaly_data}", json.dumps(anomalies, ensure_ascii=False)
            ),
        )
        level = decision.get("alert", {}).get("level", "unknown")
        print(f"  最终告警级别: {level}")
        print(f"  告警标题: {decision.get('alert', {}).get('title')}")
    except Exception as e:
        print(f"  [XX] 告警判定失败: {e}")
        return

    step_header("Step 5: 告警摘要生成")
    try:
        summary = client.chat_json(
            user_message=json.dumps(decision, ensure_ascii=False),
            system_prompt=ALERT_SUMMARY_PROMPT.replace(
                "{alert_decision}", json.dumps(decision, ensure_ascii=False)
            ),
        )
        print("  [WS] WebSocket 推送:")
        print(f"    {json.dumps(summary.get('websocket', {}), ensure_ascii=False, indent=4)}")
        print("\n  [Webhook] (飞书/钉钉):")
        print(f"    {summary.get('webhook_markdown', 'N/A')[:200]}")
        print("\n  [Log] 日志记录:")
        print(f"    {summary.get('log_entry', 'N/A')}")
    except Exception as e:
        print(f"  [XX] 摘要生成失败: {e}")

    print(f"\n  [OK] 完整告警流水线验证通过!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="告警 Agent 演示")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="干跑模式：仅展示 Prompt 模板填充效果，不实际调用 LLM API",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="实际调用 LLM API（需先设置 API Key）",
    )
    parser.add_argument(
        "--provider",
        default="deepseek",
        choices=["deepseek", "kimi", "openai"],
        help="LLM Provider（默认: deepseek）",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  告警 Agent 演示 — Day 1: 调研 LLM API + Prompt 模板")
    print("=" * 60)

    if args.live:
        demo_live(args.provider)
    else:
        demo_dry_run()
        print("\n[Tip] 如果已配置 LLM API Key，可用 --live 参数实际调用:")
        print(f"   python -m alert_agent.demo --live --provider {args.provider}")
