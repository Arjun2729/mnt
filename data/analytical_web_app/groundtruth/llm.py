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
        # Lite models get far more generous free-tier quotas than the flagship ones.
        # Measured against a free key: gemini-3.6-flash allows 5 requests/minute, while
        # 2.5-flash-lite absorbed a 14-request burst without throttling. Since the
        # analyst spends one request per tool round, that difference decides whether a
        # single question completes.
        default_model="gemini-2.5-flash-lite",
        key_env="GEMINI_API_KEY",
        signup_url="https://aistudio.google.com/apikey",
        free=True,
        note="Free tier, no payment method required. Supports tool calling and streaming.",
        # Verified reachable and tool-calling capable on a free key. Ids do get retired —
        # `list_models` reports what yours can reach today.
        suggested_models=(
            "gemini-2.5-flash-lite",
            "gemini-flash-lite-latest",
            "gemini-3.1-flash-lite",
            "gemini-2.5-flash",
            "gemini-3.6-flash",
        ),
    ),
    "Groq": Provider(
        name="Groq",
        base_url="https://api.groq.com/openai/v1",
        # Verified against a live free key: these three call tools correctly.
        # Groq rotates its catalogue, so use "List models" if one 404s.
        default_model="openai/gpt-oss-120b",
        key_env="GROQ_API_KEY",
        signup_url="https://console.groq.com/keys",
        free=True,
        note="Free tier, unusually fast, and generous enough for an agent that spends "
             "several requests per question. Tool-calling support varies by model — "
             "`groq/compound` does not support it.",
        suggested_models=("openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.6-27b"),
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

# Groq leads: its free tier is the only one that comfortably carries an agent
# spending one request per tool round. The others exist because model ids and
# quotas move without notice — three of them broke during development — and
# switching provider should cost a dropdown, not a rewrite.
# Present the recommended provider first.
PROVIDERS = {"Groq": PROVIDERS["Groq"], **{k: v for k, v in PROVIDERS.items() if k != "Groq"}}

DEFAULT_PROVIDER = "Groq"


def resolve_provider(name: str | None = None) -> Provider:
    chosen = name or os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER
    return PROVIDERS.get(chosen, PROVIDERS[DEFAULT_PROVIDER])


def resolve_key(provider: Provider) -> str:
    """Provider-specific variable first, then a generic fallback."""
    return os.getenv(provider.key_env) or os.getenv("LLM_API_KEY") or ""


def resolve_model(provider: Provider, use_env: bool = True) -> str:
    """The model to start with for a provider.

    LLM_MODEL names a model for the configured provider, so it must not follow the
    user to a different one — a Gemini id sent to Groq is simply wrong. Callers
    switching provider pass use_env=False to get that provider's own default.
    """
    if use_env:
        override = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL")
        if override:
            return override
    return provider.default_model


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
        # Reasoning models can spend a small token budget before emitting text, so
        # an empty body still means the endpoint, key and model are all working.
        detail = f"replied {reply!r}" if reply else "responded (no text within the token budget)"
        return True, f"Connected to {model} — {detail}"
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
    message = str(error)
    match = re.search(r"limit:\s*(\d+)", message) or re.search(r"'quotaValue':\s*'?(\d+)", message)
    return match.group(1) if match else None


def parse_quota_scope(error: Exception | str) -> str | None:
    """Whether an exhausted quota is per-minute or per-day.

    The distinction decides whether waiting helps. A daily quota still reports a
    short retryDelay, so retrying on that advice burns attempts against a budget
    that will not refill for hours.
    """
    message = str(error)
    if re.search(r"PerDay|per[-_ ]?day|daily", message, re.IGNORECASE):
        return "day"
    if re.search(r"PerMinute|per[-_ ]?minute", message, re.IGNORECASE):
        return "minute"
    return None


def is_retryable_rate_limit(error: Exception | str) -> bool:
    """Only a per-minute ceiling is worth waiting out."""
    return is_rate_limited(error) and parse_quota_scope(error) != "day"


# Free tiers are metered per minute, and one analyst question costs one request
# per tool round — so a 5/min ceiling is roughly one question. These notes are
# guidance rather than a contract; providers change quotas without warning.
RATE_LIMIT_ADVICE: dict[str, str] = {
    "Google Gemini": (
        "The free tier is metered **per day as well as per minute**, and the daily budget is "
        "the binding one: measured on a free key, gemini-2.5-flash-lite allows 20 requests "
        "per day. Since the analyst spends one request per tool round, that is roughly four "
        "to six questions in total. Workable for a look around; not for real use. "
        "**Groq's free tier is the better choice here**, or run Ollama locally."
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
