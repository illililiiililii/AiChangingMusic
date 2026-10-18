"""Fetch the pinned upstream RVC assets required for the local 32 kHz proof."""

import hashlib
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download


REPO = "lj1995/VoiceConversionWebUI"
REVISION = "e6d0c1a17da07c33557852f9dfa2bd44cc75737d"
ASSETS = (
    "hubert_base/config.json",
    "hubert_base/preprocessor_config.json",
    "hubert_base/pytorch_model.bin",
    "pretrained_v2/f0G32k.pth",
    "pretrained_v2/f0D32k.pth",
    "rmvpe.pt",
    "mute.zip",
)
SHA256 = {
    "hubert_base/config.json": "0346950779dfb7f9316fa74ed846e2b8a22a08eedfdc5387b73f327cb1a4a7cf",
    "hubert_base/preprocessor_config.json": "7c1976a680fb7acc757cd36fb08eef878fa36c70b4c9d2d595df9c608bbbbf0e",
    "hubert_base/pytorch_model.bin": "cc8c20f4b90a520757260197a3ff2505705a7adbd20ad9eeaa4e1a9b38442ef5",
    "pretrained_v2/f0G32k.pth": "2332611297b8d88c7436de8f17ef5f07a2119353e962cd93cda5806d59a1133d",
    "pretrained_v2/f0D32k.pth": "bd7134e7793674c85474d5145d2d982e3c5d8124fc7bb6c20f710ed65808fa8a",
    "rmvpe.pt": "6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193",
    "mute.zip": "ee948e85213e4ed2f2ba2f8dfcee810bfd0b63131d91450e920bbe1cbd0321d0",
}


def verify(path, name):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != SHA256[name]:
        raise RuntimeError(f"SHA256 mismatch: {name}")


def main():
    root = Path(__file__).resolve().parent.parent / "external" / "RVC"
    asset_root = root / "assets"
    for name in ASSETS:
        print(f"Downloading {name}", flush=True)
        destination = asset_root / "rmvpe" / "rmvpe.pt" if name == "rmvpe.pt" else asset_root / name
        if destination.is_file():
            verify(destination, name)
            continue
        path = Path(hf_hub_download(REPO, name, revision=REVISION,
                                    local_dir=asset_root))
        verify(path, name)
    rmvpe_dir = asset_root / "rmvpe"
    rmvpe_dir.mkdir(exist_ok=True)
    if not (rmvpe_dir / "rmvpe.pt").exists():
        shutil.move(str(asset_root / "rmvpe.pt"), str(rmvpe_dir / "rmvpe.pt"))
    mute_zip = asset_root / "mute.zip"
    with zipfile.ZipFile(mute_zip) as archive:
        for entry in archive.infolist():
            target = (root / "logs" / entry.filename).resolve()
            if not target.is_relative_to((root / "logs").resolve()):
                raise RuntimeError(f"Invalid archive path: {entry.filename}")
        archive.extractall(root / "logs")
    print("RVC assets ready", flush=True)


if __name__ == "__main__":
    main()
