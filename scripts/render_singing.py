"""Render an editable project score using the locally installed DiffSingerKR."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

root = Path(__file__).resolve().parents[1]
model = root / "external" / "DiffSingerKR-space"


def chunks(events: list[dict]) -> list[list[dict]]:
    result = []
    current = []
    for event in events:
        if current and (event["start"] - current[-1]["end"] > .25 or len(current) >= 24):
            result.append(current)
            current = []
        current.append(event)
    if current:
        result.append(current)
    return result


def phrase_inputs(phrase: list[dict]) -> tuple[list[float], list[str], list[int]]:
    """Keep note durations and the short rests between syllables on the score timeline."""
    durations, lyrics, notes = [], [], []
    previous_end = phrase[0]["start"]
    for event in phrase:
        gap = float(event["start"] - previous_end)
        if gap < -.001:
            raise ValueError("음절 시간이 겹칩니다. 음절 타이밍을 확인하세요.")
        if gap > .001:
            durations.append(gap)
            lyrics.append("<X>")
            notes.append(0)
        duration = float(event["end"] - event["start"])
        if duration <= 0:
            raise ValueError("음절 길이는 0보다 커야 합니다. 음절 타이밍을 확인하세요.")
        durations.append(duration)
        lyrics.append(event["syllable"])
        notes.append(int(event["midi"]) + 12)  # checkpoint sings ~one octave below input note IDs
        previous_end = event["end"]
    return durations, lyrics, notes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("score", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--start", type=float)
    parser.add_argument("--end", type=float)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.cpu else str(args.gpu_index)
    if not (model / "Checkpoint" / "S_200000.pt").is_file():
        raise FileNotFoundError("DiffSingerKR 모델이 없습니다. scripts/fetch_diffsingerkr.py를 실행하세요.")
    data = json.loads(args.score.read_text(encoding="utf-8"))
    all_events = data["syllables"]
    events = all_events
    if args.start is not None and args.end is not None:
        events = [e for e in all_events if e["end"] > args.start and e["start"] < args.end]
        if not events:
            raise ValueError("선택 구간에 음절이 없습니다.")
    if not events or any(len(e["syllable"]) != 1 or not "가" <= e["syllable"] <= "힣" for e in events):
        raise ValueError("현재 DiffSingerKR 합성은 한글 음절만 지원합니다. 영어는 한국어 발음으로 고쳐 주세요.")
    sys.path.insert(0, str(model))
    os.chdir(model)
    from Inference import Inferencer  # noqa: E402

    inferencer = Inferencer(hp_path="Hyper_Parameters.yaml", checkpoint_path="Checkpoint/S_200000.pt", batch_size=1)
    sr = int(inferencer.hp.Sound.Sample_Rate)
    target_length = round(max(e["end"] for e in all_events) * sr)
    if args.start is not None and args.output.is_file():
        old, old_rate = sf.read(args.output, dtype="float32")
        if old_rate != sr or len(old) != target_length:
            raise ValueError("기존 보컬의 길이 또는 샘플레이트가 점수와 다릅니다. 전체 합성을 다시 실행하세요.")
        output = old.copy()
        left = max(0, round(min(e["start"] for e in events) * sr))
        right = min(len(output), round(max(e["end"] for e in events) * sr))
        output[left:right] = 0
    else:
        output = np.zeros(target_length, dtype=np.float32)
    for phrase in chunks(events):
        durations, lyrics, notes = phrase_inputs(phrase)
        samples = inferencer.Inference_Epoch(message_times_list=[durations],
                    lyrics=[lyrics], notes=[notes],
                    singers=["CSD"], genres=["Children"], ddim_steps=20, use_tqdm=False)[0]
        samples = np.asarray(samples, dtype=np.float32)
        if not np.isfinite(samples).all() or not samples.size:
            raise RuntimeError("노래 합성 결과가 비어 있거나 비정상입니다.")
        if np.max(np.abs(samples)) > 1.5:
            samples /= 32768
        if np.max(np.abs(samples)) > 1.0:
            samples /= np.max(np.abs(samples)) / .98
        start = round(phrase[0]["start"] * sr)
        desired = round((phrase[-1]["end"] - phrase[0]["start"]) * sr)
        if desired < 1:
            continue
        if len(samples) != desired:
            from scipy.signal import resample
            samples = resample(samples, desired).astype(np.float32)
        end = min(start + len(samples), len(output))
        output[start:end] += samples[:end - start]
    peak = float(np.max(np.abs(output))) if output.size else 0
    if peak > .98:
        output *= .98 / peak
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, output, sr, subtype="PCM_16")
    print(json.dumps({"output": str(args.output), "seconds": len(output) / sr,
                      "phrases": len(chunks(events)), "peak": peak}, ensure_ascii=False))


if __name__ == "__main__":
    main()
