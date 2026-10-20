"""Run one stage from a saved project without opening the desktop UI."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.core.project import STAGES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    parser.add_argument("stage", choices=STAGES)
    args = parser.parse_args()
    from app.core.crash_log import install
    install(f"stage={args.stage} project={args.project}")
    from app.core.pipeline import run_stage
    print(run_stage(args.project, args.stage, print))


if __name__ == "__main__":
    main()
