"""Download optional Demucs and faster-whisper models before offline work."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--whisper", action="store_true")
    parser.add_argument("--demucs", action="store_true")
    args = parser.parse_args()
    if not args.whisper and not args.demucs:
        args.whisper = args.demucs = True
    if args.whisper:
        from huggingface_hub import snapshot_download
        directory = ROOT / "models" / "faster-whisper-base"
        snapshot_download(repo_id="Systran/faster-whisper-base", local_dir=str(directory))
        print(f"Whisper: {directory}")
    if args.demucs:
        import torch
        from demucs.pretrained import get_model
        os.environ["TORCH_HOME"] = str(ROOT / "models" / "torch")
        get_model("htdemucs")
        print(f"Demucs cache: {os.environ['TORCH_HOME']}")


if __name__ == "__main__":
    main()
