from backend.services.llm_service import LLMService


def test_llm_fallback_without_key(monkeypatch):
    service = LLMService()
    service.api_key = ""
    assert service.summarize(["camera disconnected"], "fallback summary") == "fallback summary"


def test_llm_fallback_on_provider_error(monkeypatch):
    class Response:
        status_code = 429

        def raise_for_status(self):
            raise AssertionError("should downgrade before raise_for_status")

    def fake_post(*args, **kwargs):
        return Response()

    service = LLMService()
    service.api_key = "test-key"
    monkeypatch.setattr("backend.services.llm_service.requests.post", fake_post)

    assert service.summarize(["llm timeout"], "rule summary") == "rule summary"
