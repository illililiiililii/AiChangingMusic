"""Keep Python and native crash diagnostics in the user-visible logs folder."""

from __future__ import annotations

import faulthandler
import os
import sys
import threading
import traceback
from datetime import datetime

from app.core.activity_log import LOG_DIR

_crash_file = None


def install(context: str) -> None:
    global _crash_file
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _crash_file = (LOG_DIR / f"crash-{os.getpid()}.log").open("a", encoding="utf-8", buffering=1)
    _crash_file.write(f"{datetime.now().astimezone().isoformat()} | {context} | pid={os.getpid()}\n")
    faulthandler.enable(file=_crash_file, all_threads=True)

    def exception_hook(exc_type, exc, tb):
        traceback.print_exception(exc_type, exc, tb, file=_crash_file)
        _crash_file.flush()
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = exception_hook

    if hasattr(threading, "excepthook"):
        previous = threading.excepthook

        def thread_hook(args):
            traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback, file=_crash_file)
            _crash_file.flush()
            previous(args)

        threading.excepthook = thread_hook
