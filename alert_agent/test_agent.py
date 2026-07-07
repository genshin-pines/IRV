"""
手动测试脚本 — Alert Agent 巡检

用法:
  # 规则引擎模式（秒出，不消耗 LLM tokens）
  python alert_agent/test_agent.py

  # LLM 模式（需先设置 API Key）
  $env:DEEPSEEK_API_KEY = "sk-xxx"
  python alert_agent/test_agent.py --llm
"""

import sys
import logging
from pathlib import Path

# 修复 Windows GBK 终端编码
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from alert_agent import LogCollector, simulate_logs, create_agent, create_client


def test_rule_mode():
    """规则引擎模式 — 0 延迟"""
    print("=" * 60)
    print("  Alert Agent 测试 — 规则引擎模式")
    print("=" * 60)

    # 1. 准备日志收集器，手动产生异常日志
    c = LogCollector(capacity=200)
    print(f"\n[1] 手动产生 8 条异常日志...")

    # 注册模块 logger
    for mod in ['plate_recognition', 'api_server', 'camera_stream', 'auth', 'gesture_recognition']:
        lgr = logging.getLogger(mod)
        lgr.setLevel(logging.DEBUG)
        lgr.addHandler(c)

    logging.getLogger('plate_recognition').warning('识别置信度偏低: 京X12345, 置信度: 0.38')
    logging.getLogger('plate_recognition').warning('识别置信度偏低: 沪Y67890, 置信度: 0.41')
    logging.getLogger('plate_recognition').warning('识别置信度偏低: 粤Z11111, 置信度: 0.35')
    logging.getLogger('api_server').warning('请求超时: POST /recognize-video, 耗时: 3500ms')
    logging.getLogger('camera_stream').error('RTSP 流中断: 桥面, 正在重连...')
    logging.getLogger('auth').warning('登录失败: user_03 (第3次)')
    logging.getLogger('auth').warning('登录失败: user_04 (第4次)')
    logging.getLogger('auth').warning('登录失败: user_05 (第5次)')
    print(f"    已产生 {len(c.get_all_logs())} 条日志")

    # 2. 创建 Agent 并巡检
    agent = create_agent(collector=c, use_llm=False)
    alerts = agent.run_once()

    print(f"\n[2] 第一轮巡检 → 发现 {len(alerts)} 条告警\n")
    for i, a in enumerate(alerts, 1):
        print(f"  {i}. {a['icon']} [{a['level'].upper():8s}] {a['title']}")
        print(f"     模块: {', '.join(a['affected_modules'])}")
        if a.get('webhook_markdown'):
            first_line = a['webhook_markdown'].split('\n')[0]
            print(f"     通知: {first_line}")

    # 3. 再制造新异常，测试第二轮实时检测
    print(f"\n[3] 制造新异常...")
    logging.getLogger('camera_stream').error('RTSP 流二次中断: 隧道摄像头, 正在重连...')
    logging.getLogger('auth').warning('登录失败: 用户 root (第6次)')

    new_alerts = agent.run_once()
    print(f"    第二轮巡检 → 发现 {len(new_alerts)} 条新告警")
    for a in new_alerts:
        print(f"    {a['icon']} [{a['level'].upper():8s}] {a['title']}")

    # 4. 状态
    s = agent.status
    print(f"\n[4] Agent 状态")
    print(f"    巡检次数: {s['cycle_count']}, 累计告警: {s['total_alerts']}, LLM: {'可用' if s['llm_available'] else '不可用'}")

    print(f"\n[OK] 规则引擎模式测试完成\n")


def test_llm_mode():
    """LLM 模式 — 完整 AI 分析链"""
    print("=" * 60)
    print("  Alert Agent 测试 — LLM 模式 (DeepSeek-V4)")
    print("=" * 60)

    # 1. 准备日志收集器，手动产生异常日志
    c = LogCollector(capacity=200)
    for mod in ['plate_recognition', 'api_server', 'camera_stream', 'auth', 'gesture_recognition']:
        lgr = logging.getLogger(mod)
        lgr.setLevel(logging.DEBUG)
        lgr.addHandler(c)

    logging.getLogger('plate_recognition').warning('识别置信度偏低: 京A12345, 置信度: 0.42')
    logging.getLogger('plate_recognition').warning('识别置信度偏低: 沪B67890, 置信度: 0.38')
    logging.getLogger('api_server').warning('请求超时: POST /recognize-video, 耗时: 3500ms')
    logging.getLogger('camera_stream').error('RTSP 流中断: 桥面, 正在重连...')
    logging.getLogger('gesture_recognition').warning('手势跳变频繁: 3帧内切换4次')
    print(f"\n[1] 已产生 {len(c.get_all_logs())} 条异常日志")

    # 2. 连接 LLM
    client = create_client("deepseek")
    print(f"[2] LLM 客户端就绪: {client.config.model}")

    # 3. 巡检
    agent = create_agent(client=client, collector=c, use_llm=True)
    print(f"\n[3] 开始 LLM 全链路巡检...")
    alerts = agent.run_once()

    if alerts:
        print(f"\n    发现 {len(alerts)} 条告警:")
        for a in alerts:
            print(f"\n    {'─' * 50}")
            print(f"    {a['icon']} [{a['level'].upper()}] {a['title']}")
            print(f"    摘要: {a.get('summary', a.get('detail', ''))[:120]}")
            if a.get('webhook_markdown'):
                print(f"    Webhook:")
                for line in a['webhook_markdown'].split('\n'):
                    print(f"      {line}")
            print(f"    AI 生成: {a.get('ai_generated', False)}")
    else:
        print(f"\n    [INFO] 系统正常，无告警")

    s = agent.status
    print(f"\n[4] Agent 状态: 巡检={s['cycle_count']}次, 告警={s['total_alerts']}条, 错误={s['error_count']}")
    print(f"\n[OK] LLM 模式测试完成\n")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--llm", action="store_true", help="启用 LLM 模式")
    args = p.parse_args()

    if args.llm:
        test_llm_mode()
    else:
        test_rule_mode()
        print("[提示] 要测试 LLM 模式: python alert_agent/test_agent.py --llm")
