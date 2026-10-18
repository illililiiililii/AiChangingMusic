"""Download the pinned Korean DiffSinger proof-of-concept model locally."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "external" / "DiffSingerKR-space"
REVISION = "26f95ac2554b6a6725706ed2a7991a62f7f714a9"
BASE = f"https://huggingface.co/spaces/codejin/diffsingerkr/resolve/{REVISION}/"
API = f"https://huggingface.co/api/spaces/codejin/diffsingerkr/tree/{REVISION}?recursive=true"
HASHES = {
    "Checkpoint/S_200000.pt": "6482992a43b8a98554e7ef9e487a381c2717c5828d564e6dfc6cac16a0e16092",
    "vocoder.pts": "5b47a5d03d744861f94ee973294317f738ccc6dc6d27bafa5d8db5ed18f95566",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(relative: str) -> None:
    target = DEST / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = HASHES.get(relative)
    if target.exists() and (not expected or sha256(target) == expected):
        print(f"ready: {relative}")
        return
    part = target.with_name(target.name + ".download")
    with urlopen(BASE + relative, timeout=120) as source, part.open("wb") as output:
        while block := source.read(1024 * 1024):
            output.write(block)
    if expected and sha256(part) != expected:
        part.unlink()
        raise RuntimeError(f"SHA-256 mismatch: {relative}")
    part.replace(target)
    print(f"downloaded: {relative}")


def main() -> None:
    with urlopen(API, timeout=30) as response:
        paths = [item["path"] for item in json.load(response) if item["type"] == "file"]
    for relative in paths:
        if relative == ".gitattributes":
            continue
        download(relative)
    # The Space imports pysptk at module load, although inference never calls it.
    # Its Windows wheel is unavailable; remove only that unused import locally.
    generator = DEST / "Pattern_Generator.py"
    source = generator.read_text(encoding="utf-8")
    source = source.replace("from pysptk.sptk import rapt\n", "")
    generator.write_text(source, encoding="utf-8")
    print("Local inference files prepared. All runtime inference remains offline.")


if __name__ == "__main__":
    main()
