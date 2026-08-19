"""Provider configuration — the analyst must not be tied to one vendor."""
import pytest

from groundtruth import llm


def test_default_provider_is_free():
    assert llm.PROVIDERS[llm.DEFAULT_PROVIDER].free


def test_every_provider_is_completely_specified():
    for name, provider in llm.PROVIDERS.items():
        assert provider.default_model or name.startswith("Custom")
        assert provider.key_env
        assert provider.base_url is not None or name == "OpenAI"


def test_free_providers_advertise_where_to_sign_up():
    for provider in llm.PROVIDERS.values():
        if provider.free and provider.needs_key:
            assert provider.signup_url.startswith("http")


def test_resolve_provider_falls_back_to_the_default():
    assert llm.resolve_provider("nope").name == llm.DEFAULT_PROVIDER
    assert llm.resolve_provider("Groq").name == "Groq"


def test_env_overrides_are_honoured(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "Groq")
    assert llm.resolve_provider().name == "Groq"
    monkeypatch.setenv("GEMINI_API_KEY", "abc123")
    assert llm.resolve_key(llm.PROVIDERS["Google Gemini"]) == "abc123"
    monkeypatch.setenv("LLM_MODEL", "custom-model")
    assert llm.resolve_model(llm.PROVIDERS["Google Gemini"]) == "custom-model"


def test_generic_key_is_a_fallback(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_API_KEY", "generic")
    assert llm.resolve_key(llm.PROVIDERS["Google Gemini"]) == "generic"


def test_missing_key_is_refused_with_a_signup_link(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="aistudio"):
        llm.make_client(llm.PROVIDERS["Google Gemini"], "")


def test_local_provider_needs_no_key():
    client = llm.make_client(llm.PROVIDERS["Ollama (local)"], "")
    assert "localhost" in str(client.base_url)


def test_client_points_at_the_provider_endpoint():
    client = llm.make_client(llm.PROVIDERS["Google Gemini"], "k")
    assert "generativelanguage.googleapis.com" in str(client.base_url)
    assert "groq.com" in str(llm.make_client(llm.PROVIDERS["Groq"], "k").base_url)


def test_base_url_can_be_overridden():
    client = llm.make_client(llm.PROVIDERS["Custom (OpenAI-compatible)"], "k", "https://my.host/v1")
    assert "my.host" in str(client.base_url)


def test_connection_failure_is_reported_not_raised():
    ok, message = llm.check_connection(
        llm.PROVIDERS["Custom (OpenAI-compatible)"], "k", "m", "http://127.0.0.1:1/v1"
    )
    assert ok is False and message


# ---------------- model retirement ----------------


def test_retirement_message_yields_the_replacement():
    """Providers name the successor in the 404; read it back rather than guessing."""
    error = (
        "Error code: 404 - [{'error': {'code': 404, 'message': 'This model "
        "models/gemini-2.0-flash is no longer available. Please update your code to use "
        "models/gemini-3.6-flash for the latest features and improvements.', "
        "'status': 'NOT_FOUND'}}]"
    )
    assert llm.suggest_replacement(error) == "gemini-3.6-flash"


def test_models_prefix_is_stripped():
    assert llm.suggest_replacement(
        "model is no longer available, please use models/foo-2.1-pro"
    ) == "foo-2.1-pro"


@pytest.mark.parametrize("message", [
    "Connection refused",
    "Error code: 401 - invalid api key",
    "Error code: 429 - rate limit exceeded",
    "",
])
def test_unrelated_errors_suggest_nothing(message):
    assert llm.suggest_replacement(message) is None


def test_exception_objects_are_accepted():
    exc = RuntimeError("This model is no longer available. Please use models/x-9.9-flash")
    assert llm.suggest_replacement(exc) == "x-9.9-flash"


def test_deprecation_wording_is_also_handled():
    assert llm.suggest_replacement("model deprecated; switch to gemini-9.9-pro") == "gemini-9.9-pro"


# ---------------- rate limits ----------------


def test_rate_limit_is_detected_from_a_real_429():
    error = (
        "Error code: 429 - [{'error': {'code': 429, 'message': 'You exceeded your current "
        "quota... Quota exceeded for metric: generate_content_free_tier_requests, limit: 5, "
        "model: gemini-3.6-flash Please retry in 51.99214101s.', "
        "'status': 'RESOURCE_EXHAUSTED', 'retryDelay': '51s'}}]"
    )
    assert llm.is_rate_limited(error)
    assert llm.parse_quota_limit(error) == "5"
    # A second of headroom over the provider's own figure.
    assert 52 <= llm.parse_retry_delay(error) <= 54


@pytest.mark.parametrize("message", ["Error code: 401 - bad key", "Connection refused", "404 not found"])
def test_other_errors_are_not_rate_limits(message):
    assert not llm.is_rate_limited(message)


def test_retry_delay_falls_back_when_unstated():
    assert llm.parse_retry_delay("429 too many requests", default=17.0) == 17.0


def test_retry_delay_is_capped():
    assert llm.parse_retry_delay("429, please retry in 9999s") <= 120.0


def test_every_provider_has_rate_limit_advice():
    for name in llm.PROVIDERS:
        if name.startswith("Custom"):
            continue
        assert name in llm.RATE_LIMIT_ADVICE


# ---------------- per-day versus per-minute quotas ----------------
#
# Gemini's free tier meters both. A daily exhaustion still reports a short
# retryDelay, so treating every 429 as retryable burns attempts against a budget
# that will not refill for hours.


DAILY_429 = (
    "Error code: 429 - [{'error': {'code': 429, 'message': 'You exceeded your current quota', "
    "'status': 'RESOURCE_EXHAUSTED', 'details': [{'violations': [{'quotaId': "
    "'GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'quotaValue': '20'}]}, "
    "{'retryDelay': '13s'}]}}]"
)

MINUTE_429 = (
    "Error code: 429 - Quota exceeded, limit: 5. quotaId: "
    "'GenerateRequestsPerMinutePerProjectPerModel-FreeTier'. Please retry in 52s."
)


def test_daily_quota_is_identified():
    assert llm.parse_quota_scope(DAILY_429) == "day"
    assert llm.parse_quota_limit(DAILY_429) == "20"


def test_per_minute_quota_is_identified():
    assert llm.parse_quota_scope(MINUTE_429) == "minute"
    assert llm.parse_quota_limit(MINUTE_429) == "5"


def test_only_per_minute_limits_are_worth_retrying():
    assert llm.is_retryable_rate_limit(MINUTE_429) is True
    assert llm.is_retryable_rate_limit(DAILY_429) is False


def test_both_are_still_rate_limits():
    assert llm.is_rate_limited(DAILY_429) and llm.is_rate_limited(MINUTE_429)


def test_unknown_scope_defaults_to_retryable():
    assert llm.is_retryable_rate_limit("429 too many requests") is True


def test_non_rate_limit_is_never_retryable():
    assert llm.is_retryable_rate_limit("401 unauthorized") is False
