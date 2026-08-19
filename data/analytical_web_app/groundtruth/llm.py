"""Provider configuration for the analyst.

The agent talks to an OpenAI-compatible chat-completions endpoint. Several
vendors expose one, including some with a genuinely free tier, so the provider
is a setting rather than a hardcoded dependency: only `base_url`, the key and
the model name change.

Nothing here is specific to a model. `agent.py` receives a client and knows
nothing about who built it.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Provider:
    name: str
    base_url: str | None            # None means the OpenAI SDK's own default
    default_model: str
    key_env: str
    signup_url: str
    free: bool = False
    needs_key: bool = True
    note: str = ""
    suggested_models: tuple[str, ...] = field(default_factory=tuple)


PROVIDERS: dict[str, Provider] = {
    "Google Gemini": Provider(
        name="Google Gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        default_model="gemini-3.6-flash",
        key_env="GEMINI_API_KEY",
        signup_url="https://aistudio.google.com/apikey",
        free=True,
        note="Free tier with no payment method required. Supports tool calling and streaming.",
        # Google retires model ids fairly aggressively. If one 404s, the error names its
        # replacement and `list_models` reports what the key can actually reach today.
        suggested_models=("gemini-3.6-flash", "gemini-3.6-pro", "gemini-2.5-flash"),
    ),
    "Groq": Provider(
        name="Groq",
        base_url="https://api.groq.com/openai/v1",
        default_model="llama-3.3-70b-versatile",
        key_env="GROQ_API_KEY",
        signup_url="https://console.groq.com/keys",
        free=True,
        note="Free tier, unusually fast. Tool-calling support varies by model.",
        suggested_models=("llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-20b"),
    ),
    "OpenRouter": Provider(
        name="OpenRouter",
        base_url="https://openrouter.ai/api/v1",
        default_model="google/gemini-2.0-flash-exp:free",
        key_env="OPENROUTER_API_KEY",
        signup_url="https://openrouter.ai/keys",
        free=True,
        note="Aggregates many providers. Model ids ending ':free' cost nothing.",
        suggested_models=("google/gemini-2.0-flash-exp:free", "meta-llama/llama-3.3-70b-instruct:free"),
    ),
    "Ollama (local)": Provider(
        name="Ollama (local)",
        base_url="http://localhost:11434/v1",
        default_model="llama3.1",
        key_env="OLLAMA_API_KEY",
        signup_url="https://ollama.com/download",
        free=True,
        needs_key=False,
        note="Runs on your machine. No key, no network, no limits — but you need the RAM, "
             "and the model must support tool calling.",
        suggested_models=("llama3.1", "qwen2.5", "mistral-nemo"),
    ),
    "OpenAI": Provider(
        name="OpenAI",
        base_url=None,
        default_model="gpt-4o-mini",
        key_env="OPENAI_API_KEY",
        signup_url="https://platform.openai.com/api-keys",
        free=False,
        note="Paid. Billing must be set up before any request succeeds.",
        suggested_models=("gpt-4o-mini", "gpt-4o"),
    ),
    "Custom (OpenAI-compatible)": Provider(
        name="Custom (OpenAI-compatible)",
        base_url="",
        default_model="",
        key_env="LLM_API_KEY",
        signup_url="",
        note="Any endpoint that implements /chat/completions with tool calling.",
    ),
}

DEFAULT_PROVIDER = "Google Gemini"


def resolve_provider(name: str | None = None) -> Provider:
    chosen = name or os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER
    return PROVIDERS.get(chosen, PROVIDERS[DEFAULT_PROVIDER])


def resolve_key(provider: Provider) -> str:
    """Provider-specific variable first, then a generic fallback."""
    return os.getenv(provider.key_env) or os.getenv("LLM_API_KEY") or ""


def resolve_model(provider: Provider) -> str:
    return os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or provider.default_model


def make_client(provider: Provider, api_key: str, base_url: str | None = None):
    """Build an OpenAI-SDK client pointed at the chosen provider."""
    from openai import OpenAI

    url = base_url if base_url is not None else provider.base_url
    if not provider.needs_key and not api_key:
        api_key = "not-needed"  # local servers still expect the header to exist
    if not api_key:
        raise ValueError(
            f"No API key. Set {provider.key_env} or paste a key. Get one at {provider.signup_url}"
        )
    return OpenAI(api_key=api_key, base_url=url or None, timeout=120.0, max_retries=2)


def list_models(provider: Provider, api_key: str, base_url: str | None = None) -> list[str]:
    """Ask the endpoint what it actually serves — model names change often."""
    client = make_client(provider, api_key, base_url)
    return sorted(model.id for model in client.models.list())


def check_connection(provider: Provider, api_key: str, model: str, base_url: str | None = None) -> tuple[bool, str]:
    """One cheap round trip, so setup problems surface here rather than mid-question."""
    try:
        client = make_client(provider, api_key, base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Reply with the single word: ready"}],
            max_tokens=10,
        )
        reply = (response.choices[0].message.content or "").strip()
        return True, f"Connected to {model} — replied {reply!r}"
    except Exception as exc:
        replacement = suggest_replacement(exc)
        if replacement:
            return False, f"{model} is retired. The provider suggests {replacement} — switch the model and retry."
        return False, f"{type(exc).__name__}: {exc}"


def supports_tools(provider: Provider, api_key: str, model: str, base_url: str | None = None) -> tuple[bool, str]:
    """Confirm the model will actually call a tool — the agent is useless otherwise."""
    probe = [{
        "type": "function",
        "function": {
            "name": "get_row_count",
            "description": "Return how many rows the dataset has.",
            "parameters": {"type": "object", "properties": {}},
        },
    }]
    try:
        client = make_client(provider, api_key, base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "How many rows are in the dataset? Use the tool."}],
            tools=probe,
            tool_choice="auto",
        )
        called = bool(getattr(response.choices[0].message, "tool_calls", None))
        return called, "Tool calling works." if called else (
            "The model answered without calling the tool. It may not support tool calling, "
            "which the analyst requires — try a different model."
        )
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------- model drift


_RETIRED_PATTERN = re.compile(
    r"(?:use|try|switch to|updated? to)\s+(?:models/)?([A-Za-z0-9][\w.:-]{2,})",
    re.IGNORECASE,
)


def suggest_replacement(error: Exception | str) -> str | None:
    """Pull the replacement model id out of a provider's retirement message.

    Providers retire model ids on their own schedule and usually name the successor
    in the 404 body ("...is no longer available. Please update your code to use
    models/x"). Reading it back beats making the user guess.
    """
    message = str(error)
    if "no longer available" not in message.lower() and "not found" not in message.lower() \
            and "deprecat" not in message.lower():
        return None
    match = _RETIRED_PATTERN.search(message)
    if not match:
        return None
    candidate = match.group(1).rstrip(".,'\")").strip()
    # Guard against grabbing a stray word rather than an id.
    return candidate if any(ch.isdigit() or ch in ".-:" for ch in candidate) else None


# ---------------------------------------------------------------- rate limits


_RETRY_DELAY_PATTERNS = (
    re.compile(r"retry in ([0-9]+(?:\.[0-9]+)?)\s*s", re.IGNORECASE),
    re.compile(r"'retryDelay':\s*'([0-9]+(?:\.[0-9]+)?)s'"),
    re.compile(r"retry[- ]after[\"':\s]+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE),
)


def is_rate_limited(error: Exception | str) -> bool:
    message = str(error)
    return "429" in message or "RESOURCE_EXHAUSTED" in message or "rate limit" in message.lower() \
        or "quota" in message.lower()


def parse_retry_delay(error: Exception | str, default: float = 30.0) -> float:
    """Seconds to wait, taken from the provider's own advice where given."""
    message = str(error)
    for pattern in _RETRY_DELAY_PATTERNS:
        match = pattern.search(message)
        if match:
            return min(float(match.group(1)) + 1.0, 120.0)
    return default


def parse_quota_limit(error: Exception | str) -> str | None:
    """The quota figure the provider quotes, for an honest error message."""
    match = re.search(r"limit:\s*(\d+)", str(error))
    return match.group(1) if match else None


# Free tiers are metered per minute, and one analyst question costs one request
# per tool round — so a 5/min ceiling is roughly one question. These notes are
# guidance rather than a contract; providers change quotas without warning.
RATE_LIMIT_ADVICE: dict[str, str] = {
    "Google Gemini": (
        "Free-tier limits are per-model and per-minute, and the larger models are the "
        "tightest. If you are hitting them, try a lighter model (names containing "
        "'flash-lite' or 'flash' rather than 'pro') — use List models to see what your "
        "key can reach. Groq's free tier is considerably more generous for this workload."
    ),
    "Groq": (
        "Generous free tier, typically tens of requests per minute plus a daily budget. "
        "A good fit for an agent that makes several calls per question."
    ),
    "OpenRouter": (
        "Free model quotas are shared and vary through the day. Adding a small credit "
        "balance raises them substantially."
    ),
    "Ollama (local)": "No rate limit at all — it runs on your machine.",
    "OpenAI": "Limits scale with account tier and spend.",
}
