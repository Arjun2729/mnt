"""Provider round trips, with the client stubbed."""
import pytest

from groundtruth import llm


class _Model:
    def __init__(self, ident):
        self.id = ident


class _Message:
    def __init__(self, content=None, tool_calls=None):
        self.content, self.tool_calls = content, tool_calls


class _Choice:
    def __init__(self, message):
        self.message = message


class _Completion:
    def __init__(self, message):
        self.choices = [_Choice(message)]


class _Client:
    def __init__(self, message=None, models=(), error=None):
        self._message, self._models, self._error = message, models, error
        self.chat = self
        self.models = self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        if self._error:
            raise self._error
        return _Completion(self._message)

    def list(self):
        return [_Model(m) for m in self._models]


@pytest.fixture
def gemini():
    return llm.PROVIDERS["Google Gemini"]


def stub(monkeypatch, client):
    monkeypatch.setattr(llm, "make_client", lambda *a, **k: client)


def test_list_models_is_sorted(monkeypatch, gemini):
    stub(monkeypatch, _Client(models=["z-model", "a-model"]))
    assert llm.list_models(gemini, "key") == ["a-model", "z-model"]


def test_connection_check_reports_the_reply(monkeypatch, gemini):
    stub(monkeypatch, _Client(message=_Message(content=" ready ")))
    ok, message = llm.check_connection(gemini, "key", "m")
    assert ok and "ready" in message


def test_connection_check_reports_failure(monkeypatch, gemini):
    stub(monkeypatch, _Client(error=RuntimeError("401 unauthorized")))
    ok, message = llm.check_connection(gemini, "key", "m")
    assert not ok and "401" in message


def test_connection_check_surfaces_a_retirement(monkeypatch, gemini):
    stub(monkeypatch, _Client(error=RuntimeError(
        "404 model is no longer available. Please use models/gemini-9.9-flash")))
    ok, message = llm.check_connection(gemini, "key", "old-model")
    assert not ok
    assert "retired" in message and "gemini-9.9-flash" in message


def test_tool_support_detected(monkeypatch, gemini):
    stub(monkeypatch, _Client(message=_Message(tool_calls=[object()])))
    ok, message = llm.supports_tools(gemini, "key", "m")
    assert ok and "works" in message


def test_tool_support_absent(monkeypatch, gemini):
    stub(monkeypatch, _Client(message=_Message(content="I cannot call tools")))
    ok, message = llm.supports_tools(gemini, "key", "m")
    assert not ok and "tool calling" in message


def test_tool_probe_reports_errors(monkeypatch, gemini):
    stub(monkeypatch, _Client(error=RuntimeError("boom")))
    ok, message = llm.supports_tools(gemini, "key", "m")
    assert not ok and "boom" in message


def test_empty_reply_is_still_a_success(monkeypatch, gemini):
    stub(monkeypatch, _Client(message=_Message(content=None)))
    ok, _ = llm.check_connection(gemini, "key", "m")
    assert ok


def test_retirement_parser_ignores_a_bare_word():
    assert llm.suggest_replacement("model is no longer available, please use something") is None
