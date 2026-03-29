from __future__ import annotations

try:
    from dotenv import find_dotenv, load_dotenv
except ImportError:  # pragma: no cover - depends on local environment
    find_dotenv = None
    load_dotenv = None


def load_project_dotenv() -> None:
    if find_dotenv is None or load_dotenv is None:
        return

    dotenv_path = find_dotenv(".env", usecwd=True)
    if not dotenv_path:
        return
    load_dotenv(dotenv_path, override=False)
