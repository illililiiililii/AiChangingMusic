"""Map Hangul syllables to an editable monophonic note score."""

from __future__ import annotations

import re

import hgtk


_DIGITS = "영일이삼사오육칠팔구"


def _number_to_korean(value: str) -> str:
    number = int(value)
    if number == 0:
        return "영"
    if number >= 10000:
        return "".join(_DIGITS[int(digit)] for digit in value)
    result = ""
    for digit, unit in zip(value.zfill(4), ("천", "백", "십", "")):
        amount = int(digit)
        if amount:
            result += ("" if amount == 1 and unit else _DIGITS[amount]) + unit
    return result


def syllables(text: str) -> list[str]:
    text = re.sub(r"(?i)BPM", "비피엠", text)
    text = re.sub(r"\d+", lambda match: _number_to_korean(match.group()), text)
    return re.findall(r"[가-힣]|[A-Za-z]+", text)


def _fit_notes(notes: list[dict], target: int) -> list[dict]:
    fitted = [dict(note) for note in notes]
    if target < len(fitted) * .8:
        return fitted[:target]
    if target - len(fitted) > max(3, round(len(fitted) * .2)):
        raise ValueError(f"가사 {target}음절과 음표 {len(notes)}개 차이가 큽니다. 가사나 음표를 확인하세요.")
    while len(fitted) < target:
        index = max(range(len(fitted)), key=lambda i: fitted[i]["end"] - fitted[i]["start"])
        note = fitted[index]
        duration = note["end"] - note["start"]
        if duration < .16:
            raise ValueError("음절을 나눌 만큼 긴 음표가 없습니다. 가사를 줄이거나 음표를 편집하세요.")
        middle = round((note["start"] + note["end"]) / 2, 3)
        fitted[index:index + 1] = [{**note, "end": middle}, {**note, "start": middle}]
    while len(fitted) > target:
        index = min(range(len(fitted) - 1), key=lambda i: (
            max(0, fitted[i + 1]["start"] - fitted[i]["end"]) * 10
            + abs(fitted[i + 1]["midi"] - fitted[i]["midi"]) * .1
            + fitted[i + 1]["end"] - fitted[i]["start"]))
        fitted[index:index + 2] = [{**fitted[index], "end": fitted[index + 1]["end"]}]
    return fitted


def align(text: str, notes: list[dict]) -> dict:
    units = syllables(text)
    if not units:
        raise ValueError("가사에 한국어 음절 또는 영어 단어가 없습니다.")
    if not notes:
        raise ValueError("음표 정보가 없습니다.")
    if any(not (len(unit) == 1 and "가" <= unit <= "힣") for unit in units):
        raise ValueError("노래 합성은 한글 음절만 지원합니다. 영어 단어는 한글 발음으로 적어 주세요.")
    fitted = _fit_notes(notes, len(units))
    events = []
    for unit, note in zip(units, fitted):
        coda = hgtk.letter.decompose(unit)[2]
        events.append({"syllable": unit, "start": round(note["start"], 3),
                       "end": round(note["end"], 3), "midi": note["midi"], "coda": coda})
    too_short = sum(e["end"] - e["start"] < (.15 if e["coda"] else .11) for e in events)
    score = max(0., 100 * (1 - .4 * too_short / len(events)))
    return {"score": score, "syllables": events, "note_count": len(notes),
            "adjusted_notes": len(units) - len(notes),
            "coverage": round(min(1.0, len(units) / len(notes)), 3),
            "partial": len(units) < len(notes) * .8,
            "suggestion": ("입력한 가사만 앞부분 음표에 정렬했습니다." if len(units) < len(notes) * .8
                           else "" if score >= 80 else "짧은 음절을 줄이고 음표 길이를 수정하세요.")}
