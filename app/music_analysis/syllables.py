"""Approximate sung-syllable times from local ASR word intervals.

Whisper provides word boundaries, not a Korean forced alignment. Keep the
provenance explicit so these estimates are never mistaken for ground truth.
"""

from __future__ import annotations

import re


def estimate_syllables(transcription: dict, notes: list[dict], lyrics: str | None = None) -> list[dict]:
    recognized = "\n".join(segment.get("text", "") for segment in transcription.get("segments", []))
    corrected = bool(lyrics and lyrics.strip() and lyrics.strip() != recognized.strip())
    if corrected:
        units = re.findall(r"[가-힣]|[A-Za-z]+", lyrics)
        intervals = [(float(word["start"]), float(word["end"]))
                     for segment in transcription.get("segments", [])
                     for word in segment.get("words", [])
                     if float(word["end"]) > float(word["start"])]
        if not intervals:
            intervals = [(float(note["start"]), float(note["end"])) for note in notes
                         if float(note["end"]) > float(note["start"])]
        if not units or not intervals:
            return []
        durations = [end - start for start, end in intervals]
        total_duration = sum(durations)
        shares = [len(units) * duration / total_duration for duration in durations]
        counts = [int(share) for share in shares]
        for index in sorted(range(len(shares)), key=lambda i: shares[i] - counts[i], reverse=True)[:len(units) - sum(counts)]:
            counts[index] += 1
        words = []
        offset = 0
        for (start, end), count in zip(intervals, counts):
            if count:
                words.append({"start": start, "end": end, "text": "".join(units[offset:offset + count])})
                offset += count
        transcription = {"segments": [{"words": words}]}
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
                                  "method": "edited lyrics over ASR intervals" if corrected else "ASR word interval interpolation"})
    return estimates
