"""
LLM API 配置

支持的 Provider 及其 API 端点:
  - DeepSeek:  https://api.deepseek.com/v1       (DeepSeek-V4，推荐)
  - Kimi:      https://api.moonshot.cn/v1         (月之暗面)
  - GPT-4o:    https://api.openai.com/v1          (OpenAI)

API Key 通过环境变量设置，避免硬编码:
  $env:DEEPSEEK_API_KEY = "sk-xxxx"
  $env:KIMI_API_KEY      = "moonshot-xxxx"
  $env:OPENAI_API_KEY    = "sk-xxxx"

也可以在创建客户端时直接传入 api_key 参数。
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class LLMConfig:
    """单个 LLM Provider 的配置"""

    # API 端点 (OpenAI 兼容格式)
    base_url: str
    # API Key
    api_key: str = ""
    # 模型名称
    model: str = ""
    # 请求参数
    temperature: float = 0.3   # 告警场景需要较低温度以保证一致性
    max_tokens: int = 4096
    timeout: int = 30          # 请求超时秒数
    # 额外 HTTP 头
    extra_headers: Dict[str, str] = field(default_factory=dict)


# ─── 预置 Provider 配置 ──────────────────────────────────────

PRESET_PROVIDERS: Dict[str, LLMConfig] = {
    "deepseek": LLMConfig(
        base_url="https://api.deepseek.com/v1",
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        model="deepseek-v4-flash",  # DeepSeek-V4-flash
        temperature=0.3,
        max_tokens=4096,
    ),
    "kimi": LLMConfig(
        base_url="https://api.moonshot.cn/v1",
        api_key=os.environ.get("KIMI_API_KEY", ""),
        model="moonshot-v1-8k",
        temperature=0.3,
        max_tokens=2048,
    ),
    "openai": LLMConfig(
        base_url="https://api.openai.com/v1",
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        model="gpt-4o",
        temperature=0.3,
        max_tokens=2048,
    ),
}


def get_config(provider: str = "deepseek", api_key: Optional[str] = None) -> LLMConfig:
    """
    获取指定 provider 的配置。

    Args:
        provider: "deepseek" | "kimi" | "openai"
        api_key: 可选的 API Key（优先于环境变量）

    Returns:
        LLMConfig 对象

    Raises:
        ValueError: provider 不支持
    """
    provider_lower = provider.lower()
    if provider_lower not in PRESET_PROVIDERS:
        raise ValueError(
            f"不支持的 provider: {provider}。"
            f"可选: {', '.join(PRESET_PROVIDERS.keys())}"
        )

    config = PRESET_PROVIDERS[provider_lower]

    # 允许代码中传入 api_key 覆盖环境变量
    if api_key:
        config.api_key = api_key

    if not config.api_key:
        env_var = f"{provider_lower.upper()}_API_KEY"
        raise ValueError(
            f"未找到 {provider} 的 API Key。"
            f"请设置环境变量 {env_var} 或传入 api_key 参数。\n"
            f"例如: $env:{env_var} = 'your-key'"
        )

    return config
