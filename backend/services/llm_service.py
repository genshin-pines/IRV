from __future__ import annotations

import logging
from typing import Sequence

import requests

from alert_agent.prompt import SUMMARY_PROMPT, SYSTEM_PROMPT
from backend.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT_SEC


logger = logging.getLogger("alert_agent")


class LLMService:
    def __init__(self) -> None:
        self.api_key = LLM_API_KEY
        self.base_url = LLM_BASE_URL.rstrip("/")
        self.model = LLM_MODEL
        if self.api_key:
            masked = self.api_key[:5] + "***" + self.api_key[-3:] if len(self.api_key) > 10 else "***"
            print(f"[LLM] 已配置 model={self.model} base_url={self.base_url} key={masked}")
            logger.info("LLM 已配置 model=%s base_url=%s key=%s", self.model, self.base_url, masked)
        else:
            print("[LLM] ⚠️ LLM_API_KEY 未配置，LLM 不可用")
            logger.warning("LLM_API_KEY 未配置，LLM 不可用")

    def summarize(self, logs: Sequence[str], fallback: str) -> str:
        if not self.api_key:
            logger.warning("LLM_API_KEY 未配置，使用规则引擎文本")
            return fallback
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": SUMMARY_PROMPT.format(logs="\n".join(logs[-20:]))},
            ],
            "temperature": 0.2,
            "max_tokens": 300,
        }
        try:
            logger.debug("LLM 请求 model=%s log_count=%d", self.model, len(logs))
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                timeout=LLM_TIMEOUT_SEC,
            )
            if response.status_code in {401, 403, 429, 500, 502, 503, 504}:
                logger.warning("LLM 调用失败 status=%s body=%s", response.status_code, response.text[:200])
                return fallback
            response.raise_for_status()
            data = response.json()
            llm_text = data["choices"][0]["message"]["content"].strip()
            if not llm_text:
                logger.warning("LLM 返回空内容，使用规则引擎文本")
                print("[LLM] ⚠️ 返回空内容，使用规则引擎文本")
                return fallback
            logger.info("LLM 成功生成摘要 len=%d preview=%s", len(llm_text), llm_text[:80])
            print(f"[LLM] ✅ 成功生成摘要 ({len(llm_text)}字符): {llm_text[:80]}...")
            return llm_text
        except (requests.Timeout, requests.RequestException, KeyError, ValueError) as exc:
            logger.warning("LLM 请求异常: %s", exc)
            return fallback


llm_service = LLMService()
