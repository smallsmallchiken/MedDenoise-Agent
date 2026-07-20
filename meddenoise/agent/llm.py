"""LLM客户端: 统一封装Anthropic兼容接口, 供智能调参与对话Agent使用."""

from __future__ import annotations

import json
import os
import re


def is_llm_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def llm_chat(system: str, messages: list[dict], max_tokens: int = 1200) -> str | None:
    """调用LLM返回文本, 失败返回None (调用方需自行回退)."""
    if not is_llm_available():
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"],
            base_url=os.environ.get("ANTHROPIC_BASE_URL") or None,
        )
        resp = client.messages.create(
            model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        return "".join(b.text for b in resp.content if b.type == "text")
    except Exception:
        return None


def extract_json(text: str) -> dict | None:
    """从LLM回复中提取JSON对象."""
    if not text:
        return None
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    raw = m.group(1) if m else None
    if raw is None:
        m = re.search(r"\{.*\}", text, re.S)
        raw = m.group(0) if m else None
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None
