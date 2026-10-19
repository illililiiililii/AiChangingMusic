"""Persistent, user-accessible application logs."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from app.core.settings import ROOT

LOG_DIR = ROOT / "logs"


def record(stage: str, status: str, detail: str = "") -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / "app.log"
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as output:
        output.write(f"[{timestamp}] {stage}: {status}\n")
        if detail:
            output.write(detail.rstrip() + "\n")
    return path
