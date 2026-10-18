"""Install a pinned portable Windows FFmpeg build inside ignored external/."""

import hashlib
import urllib.request
import zipfile
from pathlib import Path


BASE = "https://www.gyan.dev/ffmpeg/builds/packages/"
NAME = "ffmpeg-8.1.2-essentials_build.zip"


def main():
    root = Path(__file__).resolve().parent.parent / "external" / "ffmpeg"
    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / NAME
    expected = urllib.request.urlopen(BASE + NAME + ".sha256", timeout=30).read().decode().split()[0]
    if not zip_path.exists():
        temporary = root / (NAME + ".part")
        try:
            urllib.request.urlretrieve(BASE + NAME, temporary)
            temporary.replace(zip_path)
        finally:
            temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    with zip_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual.lower() != expected.lower():
        raise RuntimeError(f"FFmpeg checksum mismatch: {actual} != {expected}")
    with zipfile.ZipFile(zip_path) as archive:
        for executable in ("ffmpeg.exe", "ffprobe.exe"):
            matching = [name for name in archive.namelist() if name.endswith("/bin/" + executable)]
            if len(matching) != 1:
                raise RuntimeError(f"Missing {executable}")
            (root / executable).write_bytes(archive.read(matching[0]))
    print(f"verified FFmpeg SHA256 {actual}")


if __name__ == "__main__":
    main()
