"""FFmpeg based input normalization and waveform preview."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from app.core.settings import ROOT

SUPPORTED = {".mp3", ".wav", ".flac", ".ogg"}


def ffmpeg() -> Path:
    local = ROOT / "external" / "ffmpeg" / "ffmpeg.exe"
    if local.is_file():
        return local
    import shutil
    found = shutil.which("ffmpeg")
    if found:
        return Path(found)
    raise FileNotFoundError("FFmpeg가 없습니다. scripts/fetch_ffmpeg.py를 실행하세요.")


def decode(source: Path, target: Path, sample_rate: int = 44100, channels: int = 2) -> Path:
    source, target = Path(source), Path(target)
    if source.suffix.lower() not in SUPPORTED or not source.is_file():
        raise ValueError("MP3, WAV, FLAC, OGG 파일을 선택하세요.")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    command = [str(ffmpeg()), "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
               "-map", "0:a:0", "-vn", "-sn", "-dn", "-ar", str(sample_rate),
               "-ac", str(channels), "-c:a", "pcm_s16le", "-f", "wav", str(temporary)]
    try:
        result = subprocess.run(command, capture_output=True, text=True, errors="replace")
        if result.returncode or not temporary.is_file() or temporary.stat().st_size <= 44:
            raise RuntimeError(f"원곡 오디오 변환 실패: {result.stderr.strip()[-1000:]}")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def waveform(path: Path, points: int = 1600) -> tuple[np.ndarray, float]:
    info = sf.info(path)
    samples, _ = sf.read(path, dtype="float32", always_2d=True)
    mono = np.mean(samples, axis=1)
    block = max(1, int(np.ceil(len(mono) / points)))
    values = np.array([np.max(np.abs(mono[i:i + block])) for i in range(0, len(mono), block)], dtype=np.float32)
    return values, info.duration
