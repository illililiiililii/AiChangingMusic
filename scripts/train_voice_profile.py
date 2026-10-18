"""Train an RVC v2 profile from authorized ordinary speech, locally."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RVC = ROOT / "external" / "RVC"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("name")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--authorized", action="store_true", required=True)
    args = parser.parse_args()
    if not args.authorized:
        raise PermissionError("명시적 음성 학습·변환 허락이 필요합니다.")
    if not args.name.replace("_", "").isalnum() or not args.name.isascii():
        raise ValueError("프로필 이름은 영문자·숫자·밑줄만 사용할 수 있습니다.")
    data = ROOT / "voice_data" / args.name
    env = os.environ.copy()
    env.update({"PATH": str(ROOT / "external" / "ffmpeg") + os.pathsep + env.get("PATH", ""),
                "CUDA_VISIBLE_DEVICES": str(args.gpu), "HF_HUB_OFFLINE": "1", "PYTHONUTF8": "1"})
    log = data / "training.log"
    data.mkdir(parents=True, exist_ok=True)

    def step(title: str, command: list[str]) -> None:
        print("=== " + title + " ===", flush=True)
        with log.open("a", encoding="utf-8") as output:
            output.write("\n=== " + title + " ===\n")
            output.flush()
            code = subprocess.call(command, cwd=RVC, env=env, stdout=output, stderr=subprocess.STDOUT)
        if code:
            raise RuntimeError(f"{title} 실패 ({code}). 로그: {log}")

    py = sys.executable
    exp = RVC / "logs" / args.name
    step("말하기 음성 전처리", [py, str(ROOT / "scripts" / "prepare_voice_data.py"),
        str(args.source), str(data)])
    exp.mkdir(parents=True, exist_ok=True)
    step("RVC 오디오 전처리", [py, "-m", "train.preprocess", str(data / "clips"), "32000", "8",
        str(exp), "False", "3.7"])
    step("RMVPE 음정 추출", [py, "-m", "train.dataset.extract_f0", "cuda", "1", "0", "0",
        str(exp), "false"])
    step("HuBERT 특징 추출", [py, "-m", "train.dataset.extract_hubert_feature", "cuda:0", "1", "0", "0",
        str(exp), "v2", "false"])
    step("학습 목록 작성", [py, str(ROOT / "scripts" / "prepare_rvc_manifest.py"), "--experiment", args.name])
    step("검색 인덱스 작성", [py, "-m", "train.train_index", args.name, "v2", "assets/indices", "8", "single"])
    step("RVC 학습", [py, "-m", "train.train", "-e", args.name, "-sr", "32k", "-f0", "1",
        "-bs", "2", "-g", "0", "-te", str(args.epochs), "-se", "5",
        "-pg", "assets/pretrained_v2/f0G32k.pth", "-pd", "assets/pretrained_v2/f0D32k.pth",
        "-l", "1", "-c", "0", "-sw", "1", "-v", "v2"])
    models = sorted((RVC / "assets" / "weights").glob(f"{args.name}_e*.pth"), key=lambda p: p.stat().st_mtime)
    indices = sorted((RVC / "assets" / "indices").glob(f"*{args.name}*.index"), key=lambda p: p.stat().st_mtime)
    if not models or not indices:
        raise RuntimeError(f"학습 결과가 없습니다. 로그: {log}")
    print(f"MODEL={models[-1]}", flush=True)
    print(f"INDEX={indices[-1]}", flush=True)


if __name__ == "__main__":
    main()
