"""Switching provider must not carry another vendor's settings across."""
from groundtruth import llm


def test_env_model_applies_to_its_own_provider(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "gemini-2.5-flash-lite")
    assert llm.resolve_model(llm.PROVIDERS["Google Gemini"]) == "gemini-2.5-flash-lite"


def test_env_model_does_not_follow_to_another_provider(monkeypatch):
    """A Gemini id sent to Groq is simply wrong."""
    monkeypatch.setenv("LLM_MODEL", "gemini-2.5-flash-lite")
    assert llm.resolve_model(llm.PROVIDERS["Groq"], use_env=False) == "llama-3.3-70b-versatile"


def test_openai_model_variable_is_still_honoured(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    assert llm.resolve_model(llm.PROVIDERS["OpenAI"]) == "gpt-4o"


def test_defaults_apply_with_no_environment(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    for provider in llm.PROVIDERS.values():
        assert llm.resolve_model(provider) == provider.default_model


def test_every_provider_default_is_in_its_own_suggestions():
    for name, provider in llm.PROVIDERS.items():
        if provider.suggested_models:
            assert provider.default_model in provider.suggested_models, name
