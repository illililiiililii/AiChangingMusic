"""Run faster-whisper strictly from a model directory already on disk."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from faster_whisper import WhisperModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("vocal", type=Path)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--language", choices=["ko", "en", "auto"], default="ko")
    args = parser.parse_args()
    vocal, model_path, output = args.vocal, args.model_path, args.output
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)
    model = WhisperModel(str(model_path), device="cpu", compute_type="int8", local_files_only=True)
    segments, info = model.transcribe(str(vocal), word_timestamps=True, beam_size=5,
                                       language=None if args.language == "auto" else args.language)
    data = {"language": info.language, "segments": []}
    for segment in segments:
        data["segments"].append({"start": segment.start, "end": segment.end,
            "text": segment.text.strip(), "words": [
                {"start": word.start, "end": word.end, "text": word.word}
                for word in segment.words or []]})
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(data['segments'])} segments; {info.language}")


if __name__ == "__main__":
    main()
