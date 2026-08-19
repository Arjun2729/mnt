"""Credential loading. .env holds real keys; .env.example must never."""
import re
from pathlib import Path

from groundtruth.config import load_env

ROOT = Path(__file__).resolve().parents[1]

# Key shapes GitHub's push protection flags, and that we must keep out of the template.
SECRET_SHAPES = re.compile(
    r"(AIza[A-Za-z0-9_-]{20,}|AQ\.[A-Za-z0-9_-]{20,}|sk-[A-Za-z0-9_-]{16,}|"
    r"gsk_[A-Za-z0-9]{20,}|sk-or-v1-[A-Za-z0-9]{20,})"
)


def test_env_example_holds_no_real_credentials():
    """A real key here is what GitHub blocked; the template must stay empty."""
    text = (ROOT / ".env.example").read_text()
    assert not SECRET_SHAPES.search(text), "a real-looking key is present in .env.example"


def test_env_example_keys_are_blank():
    for line in (ROOT / ".env.example").read_text().splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.endswith("_API_KEY"):
            assert value.strip() == "", f"{key} should be blank in the template"


def test_real_env_is_gitignored():
    ignored = (ROOT.parent.parent / ".gitignore").read_text()
    assert "\n.env\n" in ignored


def test_load_env_reads_values(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("GT_TEST_KEY=abc123\n# comment\n\nGT_QUOTED='quoted'\n")
    monkeypatch.delenv("GT_TEST_KEY", raising=False)
    assert load_env(env) is True
    import os
    assert os.environ["GT_TEST_KEY"] == "abc123"
    assert os.environ["GT_QUOTED"] == "quoted"


def test_shell_exports_win_over_the_file(tmp_path, monkeypatch):
    """A deliberate export should not be clobbered by a stale file."""
    import os
    env = tmp_path / ".env"
    env.write_text("GT_PRECEDENCE=from_file\n")
    monkeypatch.setenv("GT_PRECEDENCE", "from_shell")
    load_env(env)
    assert os.environ["GT_PRECEDENCE"] == "from_shell"


def test_missing_file_is_not_an_error(tmp_path):
    assert load_env(tmp_path / "nope.env") is False
