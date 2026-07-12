from alert_agent.rules import RuleEngine


def test_rule_engine_smoke():
    results = RuleEngine().analyze([
        {"module": "camera", "level": "ERROR", "message": "RTSP disconnected"},
    ])
    assert results
