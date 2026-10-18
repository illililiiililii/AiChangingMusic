"""Create RVC v2 32 kHz single-speaker training config and filelist."""

import json
import random
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="rvc_proof")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent / "external" / "RVC"
    exp = root / "logs" / args.experiment
    (root / "assets" / "weights").mkdir(parents=True, exist_ok=True)
    directories = {
        "wav": exp / "0_gt_wavs",
        "feature": exp / "3_feature768",
        "f0": exp / "2a_f0",
        "f0nsf": exp / "2b-f0nsf",
    }
    names = sorted(
        path.stem for path in directories["wav"].glob("*.wav")
        if (directories["feature"] / f"{path.stem}.npy").is_file()
        and (directories["f0"] / f"{path.name}.npy").is_file()
        and (directories["f0nsf"] / f"{path.name}.npy").is_file()
    )
    if len(names) < 50:
        raise RuntimeError(f"Too few complete RVC records: {len(names)}")
    lines = []
    for name in names:
        paths = (
            directories["wav"] / f"{name}.wav",
            directories["feature"] / f"{name}.npy",
            directories["f0"] / f"{name}.wav.npy",
            directories["f0nsf"] / f"{name}.wav.npy",
        )
        lines.append("|".join(str(path.as_posix()) for path in paths) + "|0")
    mute = root / "logs" / "mute"
    mute_paths = (
        mute / "0_gt_wavs" / "mute32k.wav",
        mute / "3_feature768" / "mute.npy",
        mute / "2a_f0" / "mute.wav.npy",
        mute / "2b-f0nsf" / "mute.wav.npy",
    )
    if not all(path.is_file() for path in mute_paths):
        raise RuntimeError("Missing mute training assets")
    lines.extend(["|".join(str(path.as_posix()) for path in mute_paths) + "|0"] * 2)
    random.Random(1234).shuffle(lines)
    (exp / "filelist.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    config = json.loads((root / "configs" / "v2" / "32k.json").read_text(encoding="utf-8"))
    (exp / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"prepared {len(names)} voiced records + 2 mute records")


if __name__ == "__main__":
    main()
