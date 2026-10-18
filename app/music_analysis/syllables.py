"""Approximate sung-syllable times from local ASR word intervals.

Whisper provides word boundaries, not a Korean forced alignment. Keep the
provenance explicit so these estimates are never mistaken for ground truth.
"""

from __future__ import annotations

import re


def estimate_syllables(transcription: dict, notes: list[dict]) -> list[dict]:
    estimates = []
    for segment in transcription.get("segments", []):
        for word in segment.get("words", []):
            units = re.findall(r"[가-힣]|[A-Za-z]+", word.get("text", ""))
            start, end = float(word["start"]), float(word["end"])
            if not units or end <= start:
                continue
            step = (end - start) / len(units)
            for index, unit in enumerate(units):
                onset = start + index * step
                offset = start + (index + 1) * step
                overlapping = [(min(offset, n["end"]) - max(onset, n["start"]), i)
                               for i, n in enumerate(notes) if n["end"] > onset and n["start"] < offset]
                note_index = max(overlapping)[1] if overlapping else None
                estimates.append({"syllable": unit, "start": round(onset, 3),
                                  "end": round(offset, 3), "note_index": note_index,
                                  "method": "ASR word interval interpolation"})
    return estimates
