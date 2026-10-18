"""Measure basic signal and pitch preservation of an RVC proof render."""

import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


def signal_stats(path):
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    return mono, sample_rate, {
        "path": str(path.resolve()),
        "sample_rate": sample_rate,
        "channels": audio.shape[1],
        "duration_seconds": round(len(mono) / sample_rate, 3),
        "peak": round(float(np.max(np.abs(mono))), 5),
        "rms": round(float(np.sqrt(np.mean(np.square(mono, dtype=np.float64)))), 5),
        "clipped_samples": int(np.count_nonzero(np.abs(mono) >= 0.999)),
    }


def f0_track(audio, sample_rate):
    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
    f0, voiced, _ = librosa.pyin(audio, sr=16000, fmin=70, fmax=1000,
                                 frame_length=2048, hop_length=256)
    times = librosa.times_like(f0, sr=16000, hop_length=256)
    return times, f0, voiced


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("converted", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    source, source_sr, source_stats = signal_stats(args.source)
    converted, converted_sr, converted_stats = signal_stats(args.converted)
    if len(converted) == 0 or converted_stats["rms"] < 0.001:
        raise RuntimeError("Converted audio is silent or empty")
    source_times, source_f0, source_voiced = f0_track(source, source_sr)
    converted_times, converted_f0, converted_voiced = f0_track(converted, converted_sr)
    nearest = np.clip(np.searchsorted(converted_times, source_times), 0, len(converted_times)-1)
    mask = source_voiced & converted_voiced[nearest]
    mask &= np.isfinite(source_f0) & np.isfinite(converted_f0[nearest])
    cents = 1200 * np.log2(converted_f0[nearest][mask] / source_f0[mask])
    result = {
        "ai_generated": True,
        "purpose": "local voice-conversion proof",
        "source": source_stats,
        "converted": converted_stats,
        "duration_difference_seconds": round(converted_stats["duration_seconds"] - source_stats["duration_seconds"], 3),
        "matched_voiced_frames": int(mask.sum()),
        "median_pitch_difference_cents": round(float(np.median(cents)), 1) if len(cents) else None,
        "median_absolute_pitch_difference_cents": round(float(np.median(np.abs(cents))), 1) if len(cents) else None,
        "p90_absolute_pitch_difference_cents": round(float(np.percentile(np.abs(cents), 90)), 1) if len(cents) else None,
        "fraction_over_50_cents": round(float(np.mean(np.abs(cents) > 50)), 4) if len(cents) else None,
    }
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
