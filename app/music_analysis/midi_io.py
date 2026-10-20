"""Import and export editable melody notes without changing their absolute times."""

from __future__ import annotations

from pathlib import Path

import mido


def read_midi(path: Path) -> tuple[list[dict], float]:
    midi = mido.MidiFile(path)
    tempo = 500000
    elapsed = 0.0
    active: dict[tuple[int, int], list[tuple[float, int]]] = {}
    notes = []
    detected_bpm = 120.0
    for message in mido.merge_tracks(midi.tracks):
        elapsed += mido.tick2second(message.time, midi.ticks_per_beat, tempo)
        if message.type == "set_tempo":
            tempo = message.tempo
            detected_bpm = round(mido.tempo2bpm(tempo), 3)
        elif message.type == "note_on" and message.velocity:
            active.setdefault((message.channel, message.note), []).append((elapsed, message.velocity))
        elif message.type in ("note_off", "note_on"):
            stack = active.get((message.channel, message.note), [])
            if stack:
                start, velocity = stack.pop(0)
                if elapsed > start:
                    notes.append({"start": round(start, 3), "end": round(elapsed, 3),
                                  "midi": message.note, "velocity": velocity})
    notes.sort(key=lambda note: (note["start"], note["midi"]))
    return notes, detected_bpm


def monophonic_draft(notes: list[dict]) -> list[dict]:
    """Remove likely overtone duplicates and trim small overlaps for a guide melody."""
    result: list[dict] = []
    for source in sorted(notes, key=lambda note: (note["start"], -note.get("velocity", 64))):
        note = dict(source)
        if note["end"] <= note["start"]:
            continue
        if result and note["start"] < result[-1]["end"]:
            previous = result[-1]
            overlap = min(previous["end"], note["end"]) - note["start"]
            shortest = min(previous["end"] - previous["start"], note["end"] - note["start"])
            if overlap > shortest * .45:
                previous_strength = previous.get("velocity", 64) * (previous["end"] - previous["start"])
                new_strength = note.get("velocity", 64) * (note["end"] - note["start"])
                if new_strength > previous_strength:
                    result.pop()
                else:
                    continue
            else:
                previous["end"] = round(note["start"], 3)
        result.append(note)
    return result


def write_midi(path: Path, notes: list[dict], bpm: float, syllables: list[str] | None = None,
               beats_per_bar: int = 4) -> None:
    if bpm <= 0:
        raise ValueError("BPM은 0보다 커야 합니다.")
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    tempo = mido.bpm2tempo(bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
    track.append(mido.MetaMessage("time_signature", numerator=beats_per_bar,
                                  denominator=4, time=0))
    events = []
    for index, note in enumerate(notes):
        start = round(mido.second2tick(float(note["start"]), midi.ticks_per_beat, tempo))
        end = max(start + 1, round(mido.second2tick(float(note["end"]), midi.ticks_per_beat, tempo)))
        pitch = int(note["midi"])
        if syllables and index < len(syllables):
            # MIDI has no declared text encoding; store UTF-8 bytes through mido's latin-1 codec.
            encoded = syllables[index].encode("utf-8").decode("latin-1")
            events.append((start, 0, mido.MetaMessage("lyrics", text=encoded)))
        events.append((start, 1, mido.Message("note_on", note=pitch, velocity=90, channel=0)))
        events.append((end, 0, mido.Message("note_off", note=pitch, velocity=0, channel=0)))
    previous = 0
    for tick, _, message in sorted(events, key=lambda item: (item[0], item[1])):
        track.append(message.copy(time=tick - previous))
        previous = tick
    midi.save(path)
