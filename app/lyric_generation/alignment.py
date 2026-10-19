"""Map Hangul syllables to an editable monophonic note score."""

from __future__ import annotations

import math
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


def _line_note_ranges(notes: list[dict], expected_counts: list[int]) -> list[tuple[int, int]]:
    """Assign every note to one lyric line, favoring rests at line boundaries."""
    count = len(notes)
    gaps = [0.] + [max(0., notes[i]["start"] - notes[i - 1]["end"])
                    for i in range(1, count)]
    # Penalize putting a long instrumental pause inside a line.
    pause_cost = [0.]
    for gap in gaps[1:]:
        pause_cost.append(pause_cost[-1] + .4 * max(0., gap - 1.5) ** 2)
    states: dict[int, tuple[float, list[int]]] = {0: (0., [])}
    for expected in expected_counts:
        following: dict[int, tuple[float, list[int]]] = {}
        for start, (cost, path) in states.items():
            for amount in range(max(1, expected - 5), expected + 6):
                end = start + amount
                if end > count:
                    continue
                difference = amount - expected
                inside = pause_cost[end - 1] - pause_cost[start]
                boundary = -1.7 * math.log1p(gaps[end] / .15) if end < count else 0.
                candidate = (cost + .65 * abs(difference) + .18 * difference ** 2
                             + inside + boundary)
                if end not in following or candidate < following[end][0]:
                    following[end] = (candidate, path + [end])
        states = following
    if count not in states:
        raise ValueError("가사 줄 수와 음표 수를 맞출 수 없습니다. 원곡 가사 또는 음표를 확인하세요.")
    ends = states[count][1]
    return list(zip([0] + ends[:-1], ends))


def _fit_line_notes(notes: list[dict], target: int) -> list[dict]:
    """Fit changed syllable counts without borrowing notes from a neighboring line."""
    fitted = [dict(note) for note in notes]
    while len(fitted) < target:
        index = max(range(len(fitted)), key=lambda i: fitted[i]["end"] - fitted[i]["start"])
        note = fitted[index]
        duration = note["end"] - note["start"]
        if duration < .16:
            raise ValueError("음절을 나눌 만큼 긴 음표가 없습니다. 해당 가사를 줄이거나 음표를 편집하세요.")
        middle = round((note["start"] + note["end"]) / 2, 3)
        fitted[index:index + 1] = [{**note, "end": middle}, {**note, "start": middle}]
    while len(fitted) > target:
        index = min(range(len(fitted) - 1), key=lambda i: (
            max(0, fitted[i + 1]["start"] - fitted[i]["end"]) * 10
            + abs(fitted[i + 1]["midi"] - fitted[i]["midi"]) * .1
            + fitted[i + 1]["end"] - fitted[i]["start"]))
        fitted[index:index + 2] = [{**fitted[index], "end": fitted[index + 1]["end"]}]
    return fitted


def align(text: str, notes: list[dict], reference_text: str = "") -> dict:
    units = syllables(text)
    if not units:
        raise ValueError("가사에 한국어 음절 또는 영어 단어가 없습니다.")
    if not notes:
        raise ValueError("음표 정보가 없습니다.")
    if any(not (len(unit) == 1 and "가" <= unit <= "힣") for unit in units):
        raise ValueError("노래 합성은 한글 음절만 지원합니다. 영어 단어는 한글 발음으로 적어 주세요.")
    lyric_lines = [syllables(line) for line in text.splitlines() if syllables(line)]
    reference_lines = [syllables(line) for line in reference_text.splitlines() if syllables(line)]
    line_ranges: list[tuple[int, int]] = []
    if len(lyric_lines) > 1 and (not reference_lines or len(reference_lines) == len(lyric_lines)):
        expected = [len(line) for line in (reference_lines or lyric_lines)]
        line_ranges = _line_note_ranges(notes, expected)
        fitted = []
        for line, (start, end) in zip(lyric_lines, line_ranges):
            fitted.extend(_fit_line_notes(notes[start:end], len(line)))
    else:
        fitted = _fit_notes(notes, len(units))
    events = []
    for unit, note in zip(units, fitted):
        coda = hgtk.letter.decompose(unit)[2]
        events.append({"syllable": unit, "start": round(note["start"], 3),
                       "end": round(note["end"], 3), "midi": note["midi"], "coda": coda})
    too_short = sum(e["end"] - e["start"] < (.15 if e["coda"] else .11) for e in events)
    score = max(0., 100 * (1 - .4 * too_short / len(events)))
    partial = not line_ranges and len(units) < len(notes) * .8
    return {"score": score, "syllables": events, "note_count": len(notes),
            "adjusted_notes": len(units) - len(notes),
            "coverage": round(min(1.0, len(units) / len(notes)), 3),
            "partial": partial, "line_note_ranges": line_ranges,
            "suggestion": ("입력한 가사만 앞부분 음표에 정렬했습니다." if partial
                           else "" if score >= 80 else "짧은 음절을 줄이고 음표 길이를 수정하세요.")}
