"""Check local tools, models, GPU, and Python environments without downloads."""

from __future__ import annotations

import importlib.util
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    packages = {name: importlib.util.find_spec(name) is not None for name in
                ("PySide6", "torch", "soundfile", "librosa")}
    report = {"python": sys.version.split()[0], "windows": platform.system() == "Windows",
              "packages": packages, "files": {
        "ffmpeg": (ROOT / "external" / "ffmpeg" / "ffmpeg.exe").is_file() or bool(shutil.which("ffmpeg")),
        "diffsinger": (ROOT / "external" / "DiffSingerKR-space" / "Checkpoint" / "S_200000.pt").is_file(),
        "korean_tts": (ROOT / "external" / "ppaso-tts" / "onnx" / "vocoder.onnx.data").is_file(),
        "rvc": (ROOT / "external" / "RVC" / "infer" / "cli.py").is_file(),
        "rvc_assets": (ROOT / "external" / "RVC" / "assets" / "hubert_base" / "pytorch_model.bin").is_file(),
        "whisper": (ROOT / "models" / "faster-whisper-base" / "model.bin").is_file(),
        "demucs": bool(list((ROOT / "models" / "torch" / "hub" / "checkpoints").glob("*.th"))),
        "rvc_python": (ROOT / ".venv-rvc" / "Scripts" / "python.exe").is_file(),
    }}
    if packages["torch"]:
        import torch
        report["torch"] = torch.__version__
        report["cuda_available"] = torch.cuda.is_available()
        report["gpus"] = [{"index": i, "name": torch.cuda.get_device_name(i),
                            "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2)}
                           for i in range(torch.cuda.device_count())]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if all(packages.values()) and all(report["files"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
