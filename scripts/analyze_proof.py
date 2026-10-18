"""Measure timing and fundamental-frequency agreement of the proof WAV."""

from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
meta_path = ROOT / "outputs" / "proof_korean_svs.json"
meta = json.loads(meta_path.read_text(encoding="utf-8"))
audio, sr = sf.read(ROOT / "outputs" / "proof_korean_svs.wav")
if audio.ndim != 1:
    raise RuntimeError("Expected mono proof WAV")
hop = 256
f0, voiced, _ = librosa.pyin(
    audio,
    sr=sr,
    fmin=120,
    fmax=1200,
    frame_length=1024,
    hop_length=hop,
)
times = librosa.times_like(f0, sr=sr, hop_length=hop)
onset = 0.0
per_note = []
for lyric, note, duration in zip(
    meta["input_syllables"], meta["target_notes_midi"], meta["input_durations_seconds"]
):
    margin = min(0.08, duration * 0.2)
    selection = (times >= onset + margin) & (times < onset + duration - margin)
    observed = f0[selection]
    observed = observed[np.isfinite(observed)]
    expected_hz = float(librosa.midi_to_hz(note))
    cents = float(1200 * np.log2(np.median(observed) / expected_hz)) if observed.size else None
    per_note.append({
        "lyric": lyric,
        "midi": note,
        "start_seconds": round(onset, 3),
        "duration_seconds": duration,
        "voiced_frames": int(observed.size),
        "pitch_error_cents": round(cents, 1) if cents is not None else None,
    })
    onset += duration

valid = [abs(row["pitch_error_cents"]) for row in per_note if row["pitch_error_cents"] is not None]
meta["pitch_review"] = {
    "method": "librosa.pyin, middle of each input note; acoustic proxy only",
    "notes_with_f0": len(valid),
    "notes_total": len(per_note),
    "median_absolute_cents": round(float(np.median(valid)), 1) if valid else None,
    "within_100_cents_fraction": round(sum(value <= 100 for value in valid) / len(valid), 3) if valid else None,
    "per_note": per_note,
}
meta["duration_error_seconds"] = round(meta["actual_seconds"] - meta["expected_seconds"], 3)
meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({
    "duration_error_seconds": meta["duration_error_seconds"],
    "pitch_review": {k: v for k, v in meta["pitch_review"].items() if k != "per_note"},
}, ensure_ascii=False, indent=2))
