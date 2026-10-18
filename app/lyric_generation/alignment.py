"""Map Hangul syllables to an editable monophonic note score."""

from __future__ import annotations

import re

import hgtk


def syllables(text: str) -> list[str]:
    return re.findall(r"[가-힣]|[A-Za-z]+|\d", text)


def align(text: str, notes: list[dict]) -> dict:
    units = syllables(text)
    if not units:
        raise ValueError("가사에 한국어 음절 또는 영어 단어가 없습니다.")
    if not notes:
        raise ValueError("음표 정보가 없습니다.")
    if len(units) == len(notes):
        events = []
        for unit, note in zip(units, notes):
            coda = hgtk.letter.decompose(unit)[2] if len(unit) == 1 and "가" <= unit <= "힣" else ""
            events.append({"syllable": unit, "start": round(note["start"], 3),
                           "end": round(note["end"], 3), "midi": note["midi"], "coda": coda})
        too_short = sum(e["end"] - e["start"] < (.15 if e["coda"] else .11) for e in events)
        score = max(0., 100 * (1 - .4 * too_short / len(events)))
        return {"score": score, "syllables": events, "note_count": len(notes),
                "suggestion": "" if score >= 80 else "짧은 음절을 줄이고 음표 길이를 수정하세요."}
    # Each score event must carry one vocalizable unit. Long notes can be split;
    # excess syllables receive a shorter share of their assigned note.
    cumulative = [0.0]
    for note in notes:
        cumulative.append(cumulative[-1] + max(.05, note["end"] - note["start"]))
    total = cumulative[-1]
    events = []
    for i, unit in enumerate(units):
        midpoint = (i + .5) * total / len(units)
        j = next((k for k in range(len(notes)) if cumulative[k + 1] >= midpoint), len(notes) - 1)
        note = notes[j]
        local_start = max(cumulative[j], i * total / len(units))
        local_end = min(cumulative[j + 1], (i + 1) * total / len(units))
        if local_end <= local_start:
            local_start, local_end = cumulative[j], cumulative[j + 1]
        ratio_a = (local_start - cumulative[j]) / (cumulative[j + 1] - cumulative[j])
        ratio_b = (local_end - cumulative[j]) / (cumulative[j + 1] - cumulative[j])
        onset = note["start"] + ratio_a * (note["end"] - note["start"])
        offset = note["start"] + ratio_b * (note["end"] - note["start"])
        coda = hgtk.letter.decompose(unit)[2] if len(unit) == 1 and "가" <= unit <= "힣" else ""
        events.append({"syllable": unit, "start": round(onset, 3), "end": round(offset, 3),
                       "midi": note["midi"], "coda": coda})
    count_penalty = abs(len(units) - len(notes)) / max(len(units), len(notes))
    too_short = sum(event["end"] - event["start"] < (.15 if event["coda"] else .11) for event in events)
    score = max(0.0, 100 * (1 - .6 * count_penalty - .4 * too_short / len(events)))
    return {"score": score, "syllables": events, "note_count": len(notes),
            "suggestion": "짧은 음절을 줄이고, JSON 점수를 직접 조정할 수 있습니다." if score < 80 else ""}
