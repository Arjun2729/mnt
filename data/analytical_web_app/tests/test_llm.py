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
