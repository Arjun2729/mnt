"""Environment loading.

Credentials live in a gitignored `.env` beside the app; `.env.example` is the
template and must never hold a real key. Loading happens once, at import of the
entry point, before anything reads os.environ.
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def load_env(path: Path | None = None, override: bool = False) -> bool:
    """Read .env into the environment. Returns whether a file was found.

    Values already exported in the shell win by default, so a deliberate
    `export` in the terminal is not silently overridden by a stale file.
    """
    target = path or ENV_PATH
    if not target.exists():
        return False
    try:
        from dotenv import load_dotenv
    except ImportError:
        # Tiny fallback so a missing optional dependency cannot break startup.
        for line in target.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip().strip("\'\"")
            if key and (override or key not in os.environ):
                os.environ[key] = value
        return True
    load_dotenv(target, override=override)
    return True
