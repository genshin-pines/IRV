import asyncio

from alert_agent.agent import AlertAgent
from alert_agent.rules import RuleResult
from backend.models.alert_event import AlertLevel
from backend.routers import alerts as alerts_router
from backend.services import llm_service


def test_llm_degradation_rule_does_not_call_llm(monkeypatch):
    def unexpected_call(*_args, **_kwargs):
        raise AssertionError("LLM degradation must not call the failing LLM again")

    monkeypatch.setattr(llm_service.llm_service, "summarize", unexpected_call)
    result = RuleResult(
        rule_id="llm_degradation",
        level=AlertLevel.ERROR,
        title="LLM 接口异常或已降级",
        summary="规则摘要",
        source_module="llm",
        raw_logs=["LLM downgraded, status=401"],
    )

    payload = AlertAgent()._to_create_payload(result)
    assert payload.summary == "规则摘要"
    assert payload.llm_summary == ""
    assert payload.ai_generated is False


def test_quick_injection_triggers_agent_immediately(monkeypatch):
    calls = []

    class FakeAgent:
        async def trigger(self):
            calls.append(True)
            return []

    monkeypatch.setattr("alert_agent.scheduler._agent", FakeAgent())
    payload = alerts_router.SimulateRequest(scenario="traffic_police_anomaly", count=1)
    result = asyncio.run(alerts_router.api_simulate_logs(payload))

    assert result["data"]["count"] == 1
    assert calls == [True]
