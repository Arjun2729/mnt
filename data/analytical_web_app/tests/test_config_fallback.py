"""The .env parser's dependency-free fallback path."""
import sys

import pytest

from groundtruth import config


@pytest.fixture
def no_dotenv(monkeypatch):
    """Force the fallback by making the optional dependency unimportable."""
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, *args, **kwargs):
        if name == "dotenv":
            raise ImportError("simulated missing dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)


def test_fallback_parses_plain_pairs(tmp_path, monkeypatch, no_dotenv):
    env = tmp_path / ".env"
    env.write_text("A=1\nB=two\n")
    monkeypatch.delenv("A", raising=False)
    monkeypatch.delenv("B", raising=False)
    assert config.load_env(env) is True
    import os
    assert os.environ["A"] == "1" and os.environ["B"] == "two"


def test_fallback_ignores_comments_and_blanks(tmp_path, monkeypatch, no_dotenv):
    env = tmp_path / ".env"
    env.write_text("# a comment\n\nC=3\n   \n")
    monkeypatch.delenv("C", raising=False)
    config.load_env(env)
    import os
    assert os.environ["C"] == "3"


def test_fallback_strips_quotes(tmp_path, monkeypatch, no_dotenv):
    env = tmp_path / ".env"
    env.write_text("D='quoted'\nE=\"double\"\n")
    for key in ("D", "E"):
        monkeypatch.delenv(key, raising=False)
    config.load_env(env)
    import os
    assert os.environ["D"] == "quoted"
    assert os.environ["E"] == "double"


def test_fallback_keeps_values_with_equals_signs(tmp_path, monkeypatch, no_dotenv):
    env = tmp_path / ".env"
    env.write_text("URL=postgres://u:p@h/db?x=1\n")
    monkeypatch.delenv("URL", raising=False)
    config.load_env(env)
    import os
    assert os.environ["URL"] == "postgres://u:p@h/db?x=1"


def test_fallback_does_not_override_the_shell(tmp_path, monkeypatch, no_dotenv):
    env = tmp_path / ".env"
    env.write_text("F=from_file\n")
    monkeypatch.setenv("F", "from_shell")
    config.load_env(env)
    import os
    assert os.environ["F"] == "from_shell"


def test_fallback_skips_malformed_lines(tmp_path, monkeypatch, no_dotenv):
    env = tmp_path / ".env"
    env.write_text("no_equals_here\nG=ok\n")
    monkeypatch.delenv("G", raising=False)
    config.load_env(env)
    import os
    assert os.environ["G"] == "ok"
