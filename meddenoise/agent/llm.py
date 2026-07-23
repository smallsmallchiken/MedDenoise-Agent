"""LLM client: Anthropic-compatible API with config, streaming, cost tracking."""

from __future__ import annotations

import json
import os
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
USAGE_FILE = ROOT / "data" / "api_usage.json"
CONFIG_FILE = ROOT / "data" / "api_config.json"

MODEL_CATALOG = [
    {"id": "gpt-5.4", "name": "GPT-5.4 (allincode)", "input_price": 0.0005, "output_price": 0.0015},
    {"id": "claude-sonnet-4-5", "name": "Claude Sonnet 4.5", "input_price": 3.0, "output_price": 15.0},
    {"id": "claude-sonnet-4-5-latest", "name": "Claude Sonnet 4.5 Latest", "input_price": 3.0, "output_price": 15.0},
    {"id": "claude-3-5-sonnet-20241022", "name": "Claude 3.5 Sonnet", "input_price": 3.0, "output_price": 15.0},
]


_DEFAULT_CONFIG = {
    "model": os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
    "base_url": os.environ.get("ANTHROPIC_BASE_URL") or None,
    "max_tokens": int(os.environ.get("ANTHROPIC_MAX_TOKENS") or 1200),
}


def _load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def list_models() -> list[dict]:
    return MODEL_CATALOG


def get_api_config() -> dict:
    cfg = _load_config()
    merged = _DEFAULT_CONFIG.copy()
    merged.update({k: v for k, v in cfg.items() if v not in (None, "")})
    try:
        merged["max_tokens"] = int(merged.get("max_tokens", 1200))
    except (TypeError, ValueError):
        merged["max_tokens"] = 1200
    return merged


def save_api_config(updates: dict) -> None:
    cfg = _load_config()
    cfg.update(updates)
    if cfg.get("base_url") in ("", None):
        cfg["base_url"] = None
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def _model_cost(model_id: str) -> tuple[float, float]:
    model_id = (model_id or "").lower()
    for m in MODEL_CATALOG:
        if m["id"].lower() == model_id:
            return m["input_price"], m["output_price"]
    if "gpt" in model_id:
        return 0.005, 0.015
    if "claude" in model_id:
        return 3.0, 15.0
    return 0.005, 0.015


def _compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    in_price, out_price = _model_cost(model)
    return round(input_tokens * in_price / 1000 + output_tokens * out_price / 1000, 6)


def _load_usage() -> dict:
    if USAGE_FILE.exists():
        try:
            return json.loads(USAGE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {"total_calls": 0, "total_input_tokens": 0, "total_output_tokens": 0,
            "total_cost": 0.0, "daily": {}, "recent": []}


def _record_usage(purpose: str, model: str, input_tokens: int, output_tokens: int,
                  latency: float, ok: bool) -> None:
    try:
        u = _load_usage()
        u["total_calls"] += 1
        u["total_input_tokens"] += input_tokens
        u["total_output_tokens"] += output_tokens
        cost = _compute_cost(model, input_tokens, output_tokens)
        u["total_cost"] = round(u.get("total_cost", 0.0) + cost, 6)
        day = date.today().isoformat()
        d = u["daily"].setdefault(day, {"calls": 0, "input_tokens": 0,
                                        "output_tokens": 0, "cost": 0.0})
        d["calls"] += 1
        d["input_tokens"] += input_tokens
        d["output_tokens"] += output_tokens
        d["cost"] = round(d.get("cost", 0.0) + cost, 6)
        u["recent"] = ([{"time": time.strftime("%Y-%m-%d %H:%M:%S"),
                         "purpose": purpose, "model": model,
                         "input_tokens": input_tokens,
                         "output_tokens": output_tokens,
                         "cost": cost, "latency_s": round(latency, 1), "ok": ok}]
                       + u.get("recent", []))[:50]
        USAGE_FILE.parent.mkdir(parents=True, exist_ok=True)
        USAGE_FILE.write_text(json.dumps(u, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass


def _summarize_period(days: int) -> dict:
    u = _load_usage()
    start = date.today() - timedelta(days=days - 1)
    calls, inp, out, cost = 0, 0, 0, 0.0
    for d, v in u.get("daily", {}).items():
        if datetime.strptime(d, "%Y-%m-%d").date() >= start:
            calls += v.get("calls", 0)
            inp += v.get("input_tokens", 0)
            out += v.get("output_tokens", 0)
            cost += v.get("cost", 0.0)
    return {"calls": calls, "input_tokens": inp, "output_tokens": out,
            "cost": round(cost, 6)}


def usage_stats() -> dict:
    u = _load_usage()
    cfg = get_api_config()
    today = u["daily"].get(date.today().isoformat(),
                           {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0})
    return {
        "model": cfg.get("model"),
        "base_url": cfg.get("base_url") or "官方默认",
        "configured": is_llm_available(),
        "max_tokens": cfg.get("max_tokens"),
        "today": today,
        "week": _summarize_period(7),
        "month": _summarize_period(30),
        "total": {"calls": u["total_calls"], "input_tokens": u["total_input_tokens"],
                  "output_tokens": u["total_output_tokens"], "cost": u.get("total_cost", 0.0)},
        "daily": u["daily"],
        "recent": u["recent"],
        "models": [m["id"] for m in MODEL_CATALOG],
    }


def check_reachability() -> dict:
    if not is_llm_available():
        return {"reachable": False, "detail": "未配置API Key"}
    cfg = get_api_config()
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


def _chat_raw(system: str, messages: list[dict], max_tokens: int | None) -> tuple[str | None, dict]:
    if not is_llm_available():
        return None, {}
    cfg = get_api_config()
    import anthropic
    client = anthropic.Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"],
        base_url=cfg.get("base_url") or None,
    )
    m_tokens = max_tokens or cfg.get("max_tokens") or 1200
    resp = client.messages.create(
        model=cfg.get("model", "claude-sonnet-4-5"),
        max_tokens=m_tokens,
        system=system,
        messages=messages,
    )
    text = "".join(b.text for b in resp.content if b.type == "text")
    usage = getattr(resp, "usage", None)
    usage_dict = {
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
    }
    return text, usage_dict


def llm_chat(system: str, messages: list[dict], max_tokens: int | None = None,
             purpose: str = "对话") -> str | None:
    start = time.time()
    text, usage = _chat_raw(system, messages, max_tokens)
    if text is not None:
        cfg = get_api_config()
        _record_usage(purpose, cfg.get("model", "unknown"), usage["input_tokens"],
                      usage["output_tokens"], time.time() - start, True)
    else:
        _record_usage(purpose, get_api_config().get("model", "unknown"), 0, 0,
                      time.time() - start, False)
    return text


def llm_chat_stream(system: str, messages: list[dict], max_tokens: int | None = None,
                    purpose: str = "对话"):
    """Generator that yields text chunks and finally a metadata dict.

    Use like:
        for chunk in llm_chat_stream(...):
            if isinstance(chunk, dict):  # final marker
                full_text, usage = chunk["text"], chunk["usage"]
            else:
                yield chunk
    """
    start = time.time()
    text, usage = _chat_raw(system, messages, max_tokens)
    if text is None:
        _record_usage(purpose, get_api_config().get("model", "unknown"), 0, 0,
                      time.time() - start, False)
        return
    chunk_size = max(1, len(text) // 80)
    for i in range(0, len(text), chunk_size):
        yield text[i:i + chunk_size]
        time.sleep(0.015)
    cfg = get_api_config()
    _record_usage(purpose, cfg.get("model", "unknown"), usage["input_tokens"],
                  usage["output_tokens"], time.time() - start, True)
    yield {"final": True, "text": text, "usage": usage, "model": cfg.get("model")}


def extract_json(text: str) -> dict | None:
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
