"""Run native audio libraries outside the Qt process so crashes stay contained."""

from __future__ import annotations

import os
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Callable

from app.core.activity_log import LOG_DIR, record
from app.core.project import Project
from app.core.settings import ROOT


def run_stage_process(path: Path, stage: str, progress: Callable[[str], None] | None = None) -> str:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    command = [sys.executable, "-m", "app.core.cli", str(path), stage]
    recent = deque(maxlen=100)
    with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                          errors="replace", bufsize=1) as process:
        assert process.stdout is not None
        for line in process.stdout:
            message = line.rstrip()
            recent.append(message)
            if progress and message and ("===" in message or "Prepared" in message
                                         or "MODEL=" in message or "INDEX=" in message
                                         or "학습 에포크:" in message):
                progress(message)
        code = process.wait()
        pid = process.pid
    project = Project.load(path)
    state = project.stages.get(stage, {})
    if code:
        detail = (f"{stage} 하위 프로세스 종료: {code:#010x} ({code})\n"
                  f"진단 로그: {LOG_DIR / f'crash-{pid}.log'}\n"
                  + "\n".join(recent))
        record(stage, "프로세스 오류", detail)
        if state.get("status") != "실패":
            project.mark(stage, "실패", f"작업 프로세스가 종료됨 ({code:#010x}); logs 확인")
        raise RuntimeError(detail[-5000:])
    if state.get("status") != "완료":
        raise RuntimeError(f"{stage} 작업이 완료 상태를 기록하지 않았습니다. logs 폴더를 확인하세요.")
    return state.get("message", "완료")
