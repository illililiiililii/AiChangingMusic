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
    if len(units) != len(notes):
        raise ValueError(f"가사 {len(units)}음절과 음표 {len(notes)}개가 다릅니다. "
                         "개사 가사를 수정하거나 음표를 편집한 뒤 다시 정렬하세요.")
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
