"""Optional LLM backend for smarter clustering and ICP generation.

Supported:
  * Anthropic Claude                 -> ANTHROPIC_API_KEY
  * Any OpenAI-compatible endpoint   -> LLM_BASE_URL (+ LLM_API_KEY)
      - Ollama (free, local):  http://localhost:11434/v1     model e.g. llama3.1
      - Groq (free tier):      https://api.groq.com/openai/v1
      - Google Gemini (free tier): https://generativelanguage.googleapis.com/v1beta/openai/
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from . import http

PRESETS = {
    "Anthropic Claude": {"provider": "anthropic", "base_url": "https://api.anthropic.com",
                         "model": "claude-sonnet-5-5"},
    "Ollama (local, free)": {"provider": "openai", "base_url": "http://localhost:11434/v1",
                             "model": "llama3.1"},
    "Groq (free tier)": {"provider": "openai", "base_url": "https://api.groq.com/openai/v1",
                         "model": "llama-3.3-70b-versatile"},
    "Google Gemini (free tier)": {"provider": "openai",
                                  "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                                  "model": "gemini-2.5-flash"},
    "Other OpenAI-compatible": {"provider": "openai", "base_url": "", "model": ""},
}


@dataclass
class LLMConfig:
    provider: str      # "anthropic" | "openai"
    model: str
    api_key: str = ""
    base_url: str = ""

    @classmethod
    def from_env(cls) -> "LLMConfig | None":
        if os.getenv("ANTHROPIC_API_KEY"):
            return cls("anthropic", os.getenv("LLM_MODEL", "claude-sonnet-5-5"),
                       os.environ["ANTHROPIC_API_KEY"], "https://api.anthropic.com")
        if os.getenv("LLM_BASE_URL"):
            return cls("openai", os.getenv("LLM_MODEL", "llama3.1"),
                       os.getenv("LLM_API_KEY", ""), os.environ["LLM_BASE_URL"])
        return None


def complete(cfg: LLMConfig, prompt: str, max_tokens: int = 8000) -> str:
    if cfg.provider == "anthropic":
        resp = http.post(f"{cfg.base_url.rstrip('/')}/v1/messages", headers={
            "x-api-key": cfg.api_key, "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }, json={"model": cfg.model, "max_tokens": max_tokens,
                 "messages": [{"role": "user", "content": prompt}]}, timeout=300)
        return "".join(b.get("text", "") for b in resp.json().get("content", []))
    headers = {"content-type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    resp = http.post(f"{cfg.base_url.rstrip('/')}/chat/completions", headers=headers, json={
        "model": cfg.model, "max_tokens": max_tokens, "temperature": 0.2,
        "messages": [{"role": "user", "content": prompt}],
    }, timeout=300)
    return resp.json()["choices"][0]["message"]["content"]


def parse_json(text: str):
    """Read a JSON value from a model reply, tolerating code fences and chatter."""
    text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        try:
            return json.loads(fence.group(1))
        except ValueError:
            pass
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if starts:
        start = min(starts)
        end = max(text.rfind("}"), text.rfind("]"))
        return json.loads(text[start:end + 1])
    raise ValueError("No JSON found in model reply")


def complete_json(cfg: LLMConfig, prompt: str, max_tokens: int = 8000):
    reply = complete(cfg, prompt + "\n\nReply with only the JSON, no other text.", max_tokens)
    try:
        return parse_json(reply)
    except ValueError:
        reply = complete(cfg, prompt + "\n\nIMPORTANT: reply with valid JSON only.", max_tokens)
        return parse_json(reply)
