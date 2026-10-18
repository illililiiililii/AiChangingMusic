"""Generate lyric drafts through a loopback Ollama server only."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from app.lyric_generation.alignment import syllables


def generate_lyrics(theme: str, style: str, original: str, notes: list[dict], model: str) -> str:
    if not theme.strip():
        raise ValueError("개사 주제를 입력하세요.")
    durations = [round(n["end"] - n["start"], 2) for n in notes]
    prompt = ("한국어 노래 개사 작사가입니다. 다음 멜로디 음표에 어울리는 한국어 가사 초안을 작성하세요. "
              "원곡 가사를 그대로 반복하지 마세요. 각 한글 글자는 하나의 음절입니다. "
              "짧은 음표에는 종성이 적은 쉬운 발음을, 긴 음표에는 늘이기 좋은 모음을 배치하세요. "
              "부연 설명 없이 개사 가사만 한 줄로 출력하세요.\n"
              f"주제: {theme}\n스타일: {style}\n원곡 인식 가사(오류 가능): {original[:800]}\n"
              f"추정 음표 수: {len(notes)}\n각 음표 길이(초): {durations[:120]}\n"
              f"목표 한글 음절 수: {len(notes)}")
    text = ""
    for attempt in range(3):
        current_prompt = prompt
        if attempt:
            current_prompt += (f"\n이전 초안: {text}\n이 초안은 {len(syllables(text))}음절입니다. "
                               f"반드시 {len(notes)}개의 한글 음절로 고쳐서 가사만 출력하세요.")
        payload = json.dumps({"model": model or "qwen3.5:4b", "prompt": current_prompt,
                              "stream": False, "think": False, "keep_alive": 0,
                              "options": {"temperature": .5, "num_predict": 256}},
                             ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=payload,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                result = json.load(response)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError("로컬 Ollama에 연결하지 못했습니다. Ollama를 실행하고 모델을 설치하세요.") from exc
        raw = result.get("response", "").strip().strip('"“”')
        candidates = [line.strip().strip('"“”') for line in raw.splitlines() if line.strip()]
        candidate = next((line for line in candidates if len(syllables(line)) >= 2), "")
        if candidate and (not text or abs(len(syllables(candidate)) - len(notes)) < abs(len(syllables(text)) - len(notes))):
            text = candidate
        if len(syllables(text)) == len(notes):
            break
    if not text:
        raise RuntimeError("Ollama가 가사를 반환하지 않았습니다.")
    return text
