"""Machine-local paths and recent project list."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
USER_STATE = Path.home() / "AppData" / "Local" / "AIParodyMusicStudio" / "settings.json"


def recent_projects() -> list[str]:
    try:
        return [p for p in json.loads(USER_STATE.read_text(encoding="utf-8")).get("recent", []) if Path(p).is_file()]
    except (OSError, ValueError, TypeError):
        return []


def remember(path: Path) -> None:
    value = str(Path(path).resolve())
    recent = [value, *[p for p in recent_projects() if p != value]][:10]
    USER_STATE.parent.mkdir(parents=True, exist_ok=True)
    USER_STATE.write_text(json.dumps({"recent": recent}, ensure_ascii=False, indent=2), encoding="utf-8")
