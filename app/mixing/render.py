"""Local stereo mixing with high-pass, light compression and peak guard."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt


def _read(path: Path, rate: int) -> np.ndarray:
    data, source_rate = sf.read(path, dtype="float32", always_2d=True)
    if source_rate != rate:
        import math
        divisor = math.gcd(source_rate, rate)
        data = resample_poly(data, rate // divisor, source_rate // divisor, axis=0)
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    return data


def render_mix(vocal_path: Path, instrumental_path: Path, output: Path,
               vocal_gain: float = 1.0, instrumental_gain: float = .8,
               reverb: float = .08, rate: int = 44100) -> dict:
    vocal = _read(vocal_path, rate)
    instrumental = _read(instrumental_path, rate)
    length = max(len(vocal), len(instrumental))
    vocal = np.pad(vocal, ((0, length - len(vocal)), (0, 0)))
    instrumental = np.pad(instrumental, ((0, length - len(instrumental)), (0, 0)))
    vocal = sosfilt(butter(2, 75, fs=rate, btype="highpass", output="sos"), vocal, axis=0)
    # Smooth peaks without moving the melody or changing timing.
    vocal = np.tanh(vocal * 1.15) / np.tanh(1.15)
    if reverb > 0:
        wet = np.zeros_like(vocal)
        for delay, gain in ((.083, .55), (.137, .34), (.211, .18)):
            samples = int(delay * rate)
            wet[samples:] += vocal[:-samples] * gain
        vocal = vocal * (1 - reverb) + wet * reverb
    mixed = vocal * vocal_gain + instrumental * instrumental_gain
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > .97:
        mixed *= .97 / peak
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, mixed, rate, subtype="PCM_16")
    return {"duration": length / rate, "peak_before_guard": peak,
            "peak_after_guard": float(np.max(np.abs(mixed)))}
