"""Render an intelligible Korean guide vocal from local TTS and score timing.

This pronunciation-first path keeps contiguous Korean words together. The result is
rhythmic speech rather than a trained singing voice; RVC supplies the target timbre.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.lyric_generation.alignment import syllables
from app.audio_processing.io import ffmpeg

MODEL = ROOT / "external" / "ppaso-tts"
SAMPLE_RATE = 22050


def score_lines(lyrics: str, events: list[dict]) -> list[tuple[list[str], list[dict]]]:
    lines = []
    position = 0
    for line in lyrics.splitlines():
        words = ["".join(syllables(word)) for word in line.split()]
        words = [word for word in words if word]
        count = sum(len(word) for word in words)
        if count:
            lines.append((words, events[position:position + count]))
            position += count
    if position != len(events) or any(
            "".join(words) != "".join(event["syllable"] for event in aligned)
            for words, aligned in lines):
        raise ValueError("가사와 저장된 음절 정렬이 다릅니다. 가사 탭에서 다시 정렬하고 저장하세요.")
    return lines


def stretch(audio: np.ndarray, seconds: float) -> np.ndarray:
    desired = max(1, round(seconds * SAMPLE_RATE))
    if len(audio) < 200:
        return np.zeros(desired, dtype=np.float32)
    tempo = len(audio) / desired
    command = [str(ffmpeg()), "-hide_banner", "-loglevel", "error", "-f", "f32le",
               "-ar", str(SAMPLE_RATE), "-ac", "1", "-i", "pipe:0", "-af",
               f"rubberband=tempo={tempo:.6f}:pitch=1:formant=preserved",
               "-f", "f32le", "pipe:1"]
    result = subprocess.run(command, input=audio.astype("float32").tobytes(),
                            capture_output=True, check=True)
    rendered = np.frombuffer(result.stdout, dtype=np.float32).copy()
    return np.pad(rendered[:desired], (0, max(0, desired - len(rendered))))


def render(project_path: Path, output_path: Path) -> dict:
    data = json.loads(project_path.read_text(encoding="utf-8"))
    score = json.loads((project_path.parent / "score.json").read_text(encoding="utf-8"))
    events = score["syllables"]
    lines = score_lines(data["parody_lyrics"], events)
    if not (MODEL / "onnx" / "vocoder.onnx.data").is_file():
        raise FileNotFoundError("한국어 발음 모델이 없습니다. setup.bat을 다시 실행하세요.")
    sys.path.insert(0, str(MODEL / "example"))
    from ppaso_tts import PpasoTTS

    tts = PpasoTTS(MODEL, backend="onnx")
    output = np.zeros(round(max(event["end"] for event in events) * SAMPLE_RATE), dtype="float32")
    phrase_count = 0
    word_count = sum(len(words) for words, _ in lines)
    for words, aligned in lines:
        first = 0
        for word in words:
            last = first + len(word)
            phrase = aligned[first:last]
            target_start, target_end = phrase[0]["start"], phrase[-1]["end"]
            audio = tts.synthesize(word).astype("float32")
            samples = stretch(audio, target_end - target_start)
            start = round(target_start * SAMPLE_RATE)
            end = min(start + len(samples), len(output))
            output[start:end] += samples[:end - start]
            phrase_count += 1
            if phrase_count % 8 == 0 or phrase_count == word_count:
                print(f"=== 한국어 발음 합성 {phrase_count}/{word_count}단어", flush=True)
            first = last
    peak = float(np.max(np.abs(output)))
    if peak > .98:
        output *= .98 / peak
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, output, SAMPLE_RATE, subtype="PCM_16")
    return {"output": str(output_path), "phrases": phrase_count,
            "seconds": len(output) / SAMPLE_RATE, "peak": peak}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("project", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(render(args.project.resolve(), args.output.resolve()), ensure_ascii=False))


if __name__ == "__main__":
    main()
