from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def iso_utc_now() -> str:
    return dt.datetime.utcnow().isoformat(sep=" ") + "Z"


def path_from_cfg(cfg: dict[str, Any], key: str) -> Path:
    return Path(cfg["paths"][key])
