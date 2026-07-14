from __future__ import annotations

import logging
from typing import Sequence

import requests

from alert_agent.prompt import SUMMARY_PROMPT, SYSTEM_PROMPT
from backend.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT_SEC
from backend.services.log_service import write_log


logger = logging.getLogger("alert_agent")


class LLMService:
    def __init__(self) -> None:
        self.api_key = LLM_API_KEY
        self.base_url = LLM_BASE_URL.rstrip("/")
        self.model = LLM_MODEL

    def summarize(self, logs: Sequence[str], fallback: str) -> str:
        if not self.api_key:
            return fallback
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": SUMMARY_PROMPT.format(logs="\n".join(logs[-20:]))},
            ],
            "temperature": 0.2,
            "max_tokens": 500,
        }
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                timeout=LLM_TIMEOUT_SEC,
            )
            if response.status_code in {401, 403, 429, 500, 502, 503, 504}:
                logger.warning("LLM downgraded, status=%s", response.status_code)
                write_log(
                    "llm", "ERROR",
                    f"LLM API 鉴权/限流异常 status={response.status_code} url={self.base_url}",
                )
                return fallback
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip() or fallback
        except requests.Timeout:
            logger.warning("LLM timeout after %ss", LLM_TIMEOUT_SEC)
            write_log("llm", "ERROR", f"LLM API 请求超时 timeout={LLM_TIMEOUT_SEC}s url={self.base_url}")
            return fallback
        except requests.RequestException as exc:
            logger.warning("LLM request failed: %s", exc)
            write_log("llm", "ERROR", f"LLM API 请求异常 {type(exc).__name__}: {exc}")
            return fallback
        except (KeyError, ValueError) as exc:
            logger.warning("LLM response parse error: %s", exc)
            return fallback


llm_service = LLMService()
