"""
统一 LLM 客户端

封装 OpenAI 兼容的 Chat Completion API，支持:
  - DeepSeek-V4
  - Kimi (moonshot-v1-8k)
  - GPT-4o

所有 provider 共用同一套 OpenAI-compatible 接口，
只需切换 base_url 和 api_key 即可。

使用示例:
    from alert_agent import create_client, LLMClient

    client = create_client("deepseek", api_key="sk-xxx")
    # 或从环境变量读取: client = create_client("deepseek")

    # 简单对话
    reply = client.chat("分析这条日志是否有异常...")

    # 结构化输出 (JSON)
    result = client.chat_json("分析日志并返回 JSON", system_prompt="...")
"""

import json
import time
import logging
from typing import Optional, Dict, Any, List

import requests

from .config import LLMConfig, get_config

logger = logging.getLogger(__name__)


class LLMClient:
    """统一的 LLM API 客户端（OpenAI 兼容接口）"""

    def __init__(self, config: LLMConfig):
        """
        Args:
            config: LLMConfig 配置对象
        """
        self.config = config
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            **config.extra_headers,
        })
        self._endpoint = f"{config.base_url.rstrip('/')}/chat/completions"

    # ── 公共 API ──────────────────────────────────────────────

    def chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        发送单轮对话，返回文本回复。

        Args:
            user_message: 用户消息
            system_prompt: 系统提示词（角色设定）
            temperature: 温度参数（默认使用配置值）
            max_tokens: 最大 token 数（默认使用配置值）

        Returns:
            LLM 的文本回复

        Raises:
            RuntimeError: API 调用失败
        """
        messages = self._build_messages(system_prompt, user_message)
        return self._call_api(messages, temperature, max_tokens)

    def chat_json(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        发送对话，要求 LLM 返回结构化 JSON。

        会在 system prompt 末尾追加 JSON 格式要求，
        并自动解析返回的 JSON。

        Args:
            user_message: 用户消息
            system_prompt: 系统提示词
            temperature: 温度（JSON 模式建议较低值 0.1~0.3）
            max_tokens: 最大 token 数

        Returns:
            解析后的 dict

        Raises:
            RuntimeError: API 调用或 JSON 解析失败
        """
        json_hint = "\n\n【重要】请只输出 JSON，不要包含 markdown 代码块标记或其他文字。"
        sp = (system_prompt or "") + json_hint
        text = self.chat(user_message, system_prompt=sp, temperature=temperature, max_tokens=max_tokens)

        try:
            return json.loads(self._extract_json(text))
        except json.JSONDecodeError as e:
            logger.error(f"JSON 解析失败: {e}\n原文: {text[:500]}")
            raise RuntimeError(f"LLM 返回的内容无法解析为 JSON: {e}")

    def chat_with_history(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        多轮对话，传入完整消息历史。

        Args:
            messages: [{"role": "system"|"user"|"assistant", "content": "..."}, ...]
            temperature: 温度参数
            max_tokens: 最大 token 数

        Returns:
            LLM 的文本回复
        """
        return self._call_api(messages, temperature, max_tokens)

    # ── 内部方法 ──────────────────────────────────────────────

    def _build_messages(
        self,
        system_prompt: Optional[str],
        user_message: str,
    ) -> List[Dict[str, str]]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})
        return messages

    def _call_api(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float],
        max_tokens: Optional[int],
    ) -> str:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
        }

        t0 = time.perf_counter()
        try:
            resp = self._session.post(
                self._endpoint,
                json=payload,
                timeout=self.config.timeout,
            )
            elapsed_ms = round((time.perf_counter() - t0) * 1000)
            logger.info(f"LLM 调用 ({self.config.model}): {elapsed_ms}ms, status={resp.status_code}")

            if resp.status_code != 200:
                raise RuntimeError(
                    f"LLM API 返回错误 {resp.status_code}: {resp.text[:500]}"
                )

            data = resp.json()
            content = data["choices"][0]["message"]["content"]

            # 记录 token 用量
            usage = data.get("usage", {})
            if usage:
                logger.info(
                    f"Token 用量: prompt={usage.get('prompt_tokens')}, "
                    f"completion={usage.get('completion_tokens')}, "
                    f"total={usage.get('total_tokens')}"
                )

            return content

        except requests.Timeout:
            raise RuntimeError(f"LLM API 超时 ({self.config.timeout}s)")
        except requests.RequestException as e:
            raise RuntimeError(f"LLM API 请求失败: {e}")

    @staticmethod
    def _extract_json(text: str) -> str:
        """从 LLM 返回的文本中提取 JSON 字符串"""
        text = text.strip()

        # 去掉可能的 markdown 代码块标记
        if text.startswith("```"):
            lines = text.split("\n")
            # 去掉首行 ```json 或 ```
            if lines[0].startswith("```"):
                lines = lines[1:]
            # 去掉末行 ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        return text.strip()


def create_client(
    provider: str = "deepseek",
    api_key: Optional[str] = None,
) -> LLMClient:
    """
    创建 LLM 客户端的便捷工厂函数。

    Args:
        provider: "deepseek" | "kimi" | "openai"
        api_key: API Key（可选，优先于环境变量）

    Returns:
        LLMClient 实例

    Example:
        client = create_client("deepseek")              # 从环境变量读 key
        client = create_client("deepseek", "sk-xxx")    # 直接传 key
    """
    config = get_config(provider, api_key)
    logger.info(f"创建 LLM 客户端: provider={provider}, model={config.model}")
    return LLMClient(config)
