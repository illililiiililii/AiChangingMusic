"""Beat and monophonic vocal pitch estimates for editable score extraction."""

from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np


def analyze_vocal(path: Path) -> dict:
    y, sr = librosa.load(str(path), sr=16000, mono=True)
    if len(y) < sr:
        raise ValueError("1초 이상의 보컬이 필요합니다.")
    hop = 256
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop, trim=False)
    bpm = float(np.asarray(tempo).reshape(-1)[0])
    f0 = librosa.yin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"),
                    sr=sr, frame_length=1024, hop_length=hop)
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop)[0][:len(f0)]
    threshold = max(0.008, float(np.quantile(rms, .10)) * .5)
    voiced = rms > threshold
    midi = np.rint(librosa.hz_to_midi(np.clip(f0, 1, None))).astype(int)
    # Median over five frames removes isolated YIN octave jumps.
    from scipy.ndimage import median_filter
    midi = median_filter(midi, size=5)
    notes = []
    start = None
    previous = None
    for i, (pitch, active) in enumerate(zip(midi, voiced)):
        if not active or not 36 <= pitch <= 90:
            if start is not None and (i - start) * hop / sr >= .09:
                notes.append({"start": round(start * hop / sr, 3), "end": round(i * hop / sr, 3),
                              "midi": int(previous)})
            start, previous = None, None
            continue
        if start is None:
            start, previous = i, int(pitch)
        elif abs(int(pitch) - previous) >= 2:
            if (i - start) * hop / sr >= .09:
                notes.append({"start": round(start * hop / sr, 3), "end": round(i * hop / sr, 3),
                              "midi": int(previous)})
            start, previous = i, int(pitch)
    if start is not None:
        notes.append({"start": round(start * hop / sr, 3), "end": round(len(f0) * hop / sr, 3),
                      "midi": int(previous)})
    # Preserve local phrases and merge microgaps to avoid excess phonemes.
    merged = []
    for note in notes:
        if merged and note["midi"] == merged[-1]["midi"] and note["start"] - merged[-1]["end"] < .08:
            merged[-1]["end"] = note["end"]
        else:
            merged.append(note)
    pitch_midi = librosa.hz_to_midi(np.clip(f0, 1, None))
    for note in merged:
        a = max(0, round(note["start"] * sr / hop))
        b = min(len(pitch_midi), round(note["end"] * sr / hop))
        region = pitch_midi[a:b]
        if len(region):
            note["pitch_variation_cents"] = round(float(np.std(region) * 100), 1)
            note["pitch_glide_semitones"] = round(float(np.median(region[-max(1, len(region)//4):]) -
                                                           np.median(region[:max(1, len(region)//4)])), 2)
    sections = []
    if merged:
        if merged[0]["start"] > .35:
            sections.append({"start": 0., "end": merged[0]["start"], "label": "인트로/무보컬 후보"})
        phrase_start = merged[0]["start"]
        for previous_note, next_note in zip(merged, merged[1:]):
            gap = next_note["start"] - previous_note["end"]
            if gap > .65:
                sections.append({"start": phrase_start, "end": previous_note["end"], "label": "보컬 구간"})
                sections.append({"start": previous_note["end"], "end": next_note["start"], "label": "간주/호흡 후보"})
                phrase_start = next_note["start"]
        sections.append({"start": phrase_start, "end": merged[-1]["end"], "label": "보컬 구간"})
        if len(y) / sr - merged[-1]["end"] > .35:
            sections.append({"start": merged[-1]["end"], "end": round(len(y)/sr, 3),
                             "label": "아웃트로/무보컬 후보"})
    breath_gaps = [{"start": a["end"], "end": b["start"]} for a, b in zip(merged, merged[1:])
                   if .18 <= b["start"] - a["end"] <= 1.5]
    return {"bpm": round(bpm, 2), "duration": round(len(y) / sr, 3),
            "beat_times": [round(float(x), 3) for x in librosa.frames_to_time(beats, sr=sr, hop_length=hop)],
            "notes": merged, "sections": sections, "breath_gaps": breath_gaps,
            "pitch_method": "librosa YIN; editable estimate; section labels are heuristic"}
