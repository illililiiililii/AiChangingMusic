"""Make a short original test mix from the neutral SVS proof and generated chords."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    source = ROOT / "outputs" / "proof_korean_svs.wav"
    if not source.is_file():
        raise FileNotFoundError("Run scripts/prove_korean_svs.py first.")
    vocal, sr = sf.read(source, dtype="float32")
    if vocal.ndim == 2:
        vocal = vocal.mean(axis=1)
    rate = 44100
    vocal = resample_poly(vocal, 2, 1)
    time = np.arange(len(vocal), dtype=np.float64) / rate
    chords = [(0, [261.63, 329.63, 392.0]), (2.8, [220.0, 261.63, 329.63]),
              (5.6, [174.61, 220.0, 261.63]), (8.4, [196.0, 246.94, 293.66])]
    backing = np.zeros_like(time)
    for start, freqs in chords:
        mask = (time >= start) & (time < start + 2.8)
        t = time[mask] - start
        envelope = np.minimum(1, t * 8) * np.minimum(1, (2.8 - t) * 4)
        for f in freqs:
            backing[mask] += .035 * np.sin(2 * np.pi * f * t) * envelope
    beats = np.arange(0, time[-1], .7)
    for beat in beats:
        start = int(beat * rate)
        end = min(len(backing), start + int(.12 * rate))
        t = np.arange(end - start) / rate
        backing[start:end] += .045 * np.sin(2 * np.pi * 90 * t) * np.exp(-35 * t)
    mix = np.stack([vocal * .7 + backing * .9, vocal * .7 + backing], axis=1).astype(np.float32)
    target = ROOT / "outputs" / "demo_source.wav"
    sf.write(target, mix, rate, subtype="PCM_16")
    print(target)


if __name__ == "__main__":
    main()
