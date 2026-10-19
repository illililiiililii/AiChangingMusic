"""Make short, denoised speech WAV clips from authorized local recordings."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--max-seconds", type=int, default=1200)
    args = parser.parse_args()
    if not args.source.exists():
        raise FileNotFoundError(args.source)
    files = sorted(p for p in (args.source.rglob("*") if args.source.is_dir() else [args.source])
                   if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not files:
        raise ValueError("말하기 오디오가 없습니다.")
    ffmpeg = ROOT / "external" / "ffmpeg" / "ffmpeg.exe"
    if not ffmpeg.is_file():
        raise FileNotFoundError(ffmpeg)
    clips = args.destination / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    total, count, records = 0.0, 0, []
    for source in files:
        if total >= args.max_seconds:
            break
        command = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-i", str(source),
                   "-af", "highpass=f=75,lowpass=f=8000,afftdn=nf=-50:nr=6",
                   "-ar", "32000", "-ac", "1", "-f", "f32le", "pipe:1"]
        raw = subprocess.run(command, check=True, capture_output=True).stdout
        audio = np.frombuffer(raw, dtype="<f4")
        if audio.size == 0:
            continue
        # Fixed 12 s windows with a 0.5 s overlap keep memory bounded for RVC.
        window, hop = 12 * 32000, 11.5 * 32000
        for start in range(0, len(audio), int(hop)):
            part = audio[start:start + window]
            seconds = len(part) / 32000
            if seconds < 3 or total + seconds > args.max_seconds + 12:
                continue
            rms = float(np.sqrt(np.mean(np.square(part, dtype=np.float64))))
            peak = float(np.max(np.abs(part)))
            if not .007 < rms < .4 or peak >= .995:
                continue
            filename = f"speech_{count:05d}.wav"
            sf.write(clips / filename, part, 32000, subtype="PCM_16")
            records.append({"file": filename, "source": source.name, "seconds": round(seconds, 2), "rms": round(rms, 4)})
            total += seconds
            count += 1
            if total >= args.max_seconds:
                break
    if total < 600 or count < 50:
        raise ValueError(f"유효한 말하기 음성이 {total:.0f}초 / {count}개입니다. 10분 이상의 깨끗한 녹음이 필요합니다.")
    args.destination.mkdir(parents=True, exist_ok=True)
    (args.destination / "selection.json").write_text(json.dumps({"seconds": total,
        "count": count, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Prepared {count} clips, {total:.1f} s")


if __name__ == "__main__":
    main()
