"""LLM客户端: 统一封装Anthropic兼容接口, 供智能调参与对话Agent使用."""

from __future__ import annotations

import json
import os
import re
import time
from datetime import date
from pathlib import Path

USAGE_FILE = Path(__file__).resolve().parents[2] / "data" / "api_usage.json"


def _load_usage() -> dict:
    if USAGE_FILE.exists():
        try:
            return json.loads(USAGE_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return {"total_calls": 0, "total_input_tokens": 0, "total_output_tokens": 0,
            "daily": {}, "recent": []}


def _record_usage(purpose: str, input_tokens: int, output_tokens: int,
                  latency: float, ok: bool) -> None:
    try:
        u = _load_usage()
        u["total_calls"] += 1
        u["total_input_tokens"] += input_tokens
        u["total_output_tokens"] += output_tokens
        day = date.today().isoformat()
        d = u["daily"].setdefault(day, {"calls": 0, "input_tokens": 0,
                                        "output_tokens": 0})
        d["calls"] += 1
        d["input_tokens"] += input_tokens
        d["output_tokens"] += output_tokens
        u["recent"] = ([{"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                         "purpose": purpose, "input_tokens": input_tokens,
                         "output_tokens": output_tokens,
                         "latency_s": round(latency, 1), "ok": ok}]
                       + u.get("recent", []))[:50]
        USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        USAGE_FILE.write_text(json.dumps(u, ensure_ascii=False, indent=1))
    except Exception:
        pass


def usage_stats() -> dict:
    u = _load_usage()
    today = u["daily"].get(date.today().isoformat(),
                           {"calls": 0, "input_tokens": 0, "output_tokens": 0})
    return {"model": os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
            "base_url": os.environ.get("ANTHROPIC_BASE_URL", "官方默认"),
            "configured": is_llm_available(),
            "today": today,
            "total": {"calls": u["total_calls"],
                      "input_tokens": u["total_input_tokens"],
                      "output_tokens": u["total_output_tokens"]},
            "daily": u["daily"], "recent": u["recent"]}


def check_reachability() -> dict:
    """发送最小请求检测模型可达性."""
    if not is_llm_available():
        return {"reachable": False, "detail": "未配置API Key"}
    start = time.time()
    reply = llm_chat("你是接口可达性探针", [{"role": "user", "content": "回复ok"}],
                     max_tokens=200, purpose="可达性检测")
    latency = round(time.time() - start, 1)
    if reply is None:
        return {"reachable": False, "latency_s": latency,
                "detail": "调用失败, 请检查网络或凭证"}
    return {"reachable": True, "latency_s": latency, "detail": "模型响应正常"}


def is_llm_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def llm_chat(system: str, messages: list[dict], max_tokens: int = 1200,
             purpose: str = "对话") -> str | None:
    """调用LLM返回文本, 失败返回None (调用方需自行回退)."""
    if not is_llm_available():
        return None
    start = time.time()
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
        usage = getattr(resp, "usage", None)
        _record_usage(purpose,
                      getattr(usage, "input_tokens", 0) or 0,
                      getattr(usage, "output_tokens", 0) or 0,
                      time.time() - start, True)
        return "".join(b.text for b in resp.content if b.type == "text")
    except Exception:
        _record_usage(purpose, 0, 0, time.time() - start, False)
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
