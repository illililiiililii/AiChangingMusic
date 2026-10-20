"""Run native audio libraries outside the Qt process so crashes stay contained."""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable
from uuid import uuid4

from app.core.activity_log import LOG_DIR, record
from app.core.project import Project
from app.core.settings import ROOT


def _stage_command(path: Path, stage: str) -> list[str]:
    return [sys.executable, "-m", "app.core.cli", str(path), stage]


def run_stage_process(path: Path, stage: str, progress: Callable[[str], None] | None = None,
                      cancel_event: threading.Event | None = None) -> str:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cancel_file = LOG_DIR / f"cancel-{uuid4().hex}.flag"
    env["AICHANGINGMUSIC_CANCEL_FILE"] = str(cancel_file)
    command = _stage_command(path, stage)
    recent = deque(maxlen=100)
    pending: queue.Queue[str | None] = queue.Queue()
    try:
        with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                              errors="replace", bufsize=1) as process:
            assert process.stdout is not None

            def read_output():
                try:
                    for line in process.stdout:
                        pending.put(line.rstrip())
                finally:
                    pending.put(None)

            threading.Thread(target=read_output, daemon=True).start()
            finished_output = False
            cancel_started = None
            force_stopped = False
            while not finished_output or process.poll() is None:
                if cancel_event is not None and cancel_event.is_set():
                    if cancel_started is None:
                        cancel_file.touch()
                        cancel_started = time.monotonic()
                        record(stage, "중단 요청", f"PID {process.pid}")
                    elif time.monotonic() - cancel_started > 8 and process.poll() is None and not force_stopped:
                        force_stopped = True
                        if os.name == "nt":
                            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                           capture_output=True, timeout=10)
                        else:
                            process.terminate()
                try:
                    message = pending.get(timeout=.25)
                except queue.Empty:
                    continue
                if message is None:
                    finished_output = True
                    continue
                recent.append(message)
                if progress and message and (message.startswith("RVC_PROGRESS ") or
                                             "===" in message or "Prepared" in message or
                                             "MODEL=" in message or "INDEX=" in message or
                                             "학습 에포크:" in message):
                    progress(message)
            code = process.wait()
            pid = process.pid
    finally:
        cancel_file.unlink(missing_ok=True)
    project = Project.load(path)
    state = project.stages.get(stage, {})
    if cancel_event is not None and cancel_event.is_set() and code:
        project.mark(stage, "중단", "사용자가 작업을 중단했습니다.")
        record(stage, "중단", f"PID {pid}; 종료 코드 {code}")
        raise InterruptedError(f"{stage} 작업을 중단했습니다.")
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
