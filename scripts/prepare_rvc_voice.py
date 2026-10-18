"""Select clean speech clips for a local, single-speaker RVC proof run.

The source directory is read only. Selection metadata and copied clips stay
under the ignored voice_data directory.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf


EXCLUDED_NAME_PARTS = ("잡음", "종소리", "음악", "노래", "효과음", "소스 공유")


def inspect(path: Path):
    info = sf.info(path)
    duration = info.duration
    if not 2.5 <= duration <= 18 or info.channels > 2:
        return None
    if any(part in path.stem for part in EXCLUDED_NAME_PARTS):
        return None
    data, _ = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
    if not (0.012 <= rms <= 0.4 and 0.06 <= peak <= 1.05):
        return None
    clipped = float(np.mean(np.abs(mono) >= 0.995))
    if clipped > 0.002:
        return None
    return {"path": path, "duration": duration, "rms": rms, "peak": peak,
            "clipped_fraction": clipped}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--seconds", type=float, default=900)
    args = parser.parse_args()
    if not args.source.is_dir():
        raise SystemExit(f"Missing source directory: {args.source}")
    candidates = []
    for path in sorted(args.source.glob("*.wav")):
        try:
            item = inspect(path)
        except (RuntimeError, ValueError, OSError):
            item = None
        if item:
            candidates.append(item)
    # Longer, readable phrases provide more stable identity than tiny interjections.
    candidates.sort(key=lambda item: (-item["duration"], item["path"].name))
    chosen, seconds = [], 0.0
    for item in candidates:
        chosen.append(item)
        seconds += item["duration"]
        if seconds >= args.seconds:
            break
    if seconds < min(args.seconds, 600):
        raise SystemExit(f"Only {seconds:.1f} seconds met the criteria")
    clips_dir = args.destination / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for number, item in enumerate(chosen):
        target = clips_dir / f"speech_{number:04d}.wav"
        shutil.copyfile(item["path"], target)
        records.append({"original_name": item["path"].name,
                        "copied_name": target.name,
                        "duration_seconds": round(item["duration"], 3),
                        "rms": round(item["rms"], 5),
                        "peak": round(item["peak"], 5)})
    summary = {"source": str(args.source), "count": len(records),
               "seconds": round(seconds, 3), "selection": records}
    (args.destination / "selection.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"selected {len(records)} clips, {seconds:.1f} seconds")


if __name__ == "__main__":
    main()
