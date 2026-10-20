"""Convert prepared WAV chunks with one RVC model load.

Runs with .venv-rvc and emits machine-readable segment events on stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


def event(kind: str, **values) -> None:
    print("RVC_EVENT " + json.dumps({"type": kind, **values}, ensure_ascii=False), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pitch", type=int, default=0)
    parser.add_argument("--index-rate", type=float, default=0.0)
    parser.add_argument("--protect", type=float, default=0.2)
    args = parser.parse_args()

    rvc_root = Path(__file__).resolve().parents[1] / "external" / "RVC"
    sys.path.insert(0, str(rvc_root))
    os.chdir(rvc_root)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ["weight_root"] = str(args.model.parent)
    os.environ.setdefault("index_root", str(rvc_root / "logs"))
    os.environ.setdefault("outside_index_root", str(rvc_root / "assets" / "indices"))
    os.environ.setdefault("rmvpe_root", str(rvc_root / "assets" / "rmvpe"))

    from infer import cli
    from infer.vc.modules import VC

    model = cli.resolve_model(str(args.model))
    speaker_count, speakers = cli.load_model_metadata(model)
    speaker_id = cli.select_speaker(speaker_count, speakers, None)
    index = cli.resolve_index(str(args.index) if args.index else None,
                              model.name, speaker_id, args.index_rate)
    config = cli.create_config()
    inputs = sorted(args.input_dir.glob("*.wav"))
    if not inputs:
        raise ValueError("변환할 구간이 없습니다.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    event("ready", total=len(inputs), device=str(config.device),
          cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
          model=str(model), index=index)
    vc = VC(config)
    vc.get_vc(model.name)
    for number, source in enumerate(inputs, 1):
        target = args.output_dir / source.name
        event("start", segment=number, total=len(inputs), input=str(source), output=str(target))
        status, result = vc.vc_single(
            speaker_id, str(source), args.pitch, "rmvpe", index,
            args.index_rate, 0, 1.0, args.protect,
        )
        print(status, flush=True)
        if not result or result[0] is None or result[1] is None:
            raise RuntimeError(f"{number}/{len(inputs)} 구간 변환 실패: {status}")
        cli.write_audio(target, result[1], result[0], "wav")
        if not target.is_file() or target.stat().st_size <= 44:
            raise RuntimeError(f"{number}/{len(inputs)} 구간 출력 파일이 없습니다: {target}")
        event("done", segment=number, total=len(inputs), output=str(target))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
