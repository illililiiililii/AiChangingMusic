"""Render an original Korean phrase from explicit MIDI notes with local DiffSingerKR."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--gpu-index", type=int, default=0, help="Physical CUDA index; 1660 SUPER is 0 on the test PC")
PARSER.add_argument("--cpu", action="store_true")
ARGS = PARSER.parse_args()
if ARGS.cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(ARGS.gpu_index)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "external" / "DiffSingerKR-space"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# New lyrics, using a public example melody only as a note/time test fixture.
LYRICS = list("오늘은맑은하늘너와함께노래해푸른바람타고멀리가볼까")
DURATIONS = [
    .52, .17, .35, .35, .35, .35, .70, .35, .35, .70,
    .35, .35, .70, .52, .17, .35, .35, .35, .35, .70,
    .35, .35, .35, .35, 1.39,
]
TARGET_NOTES = [
    64, 62, 60, 62, 64, 64, 64, 62, 62, 62,
    64, 67, 67, 64, 62, 60, 62, 64, 64, 64,
    62, 62, 64, 62, 60,
]
# The released checkpoint consistently sings one octave below its note IDs.
# Keep the public API in standard MIDI pitch and compensate only at this adapter.
MODEL_NOTES = [note + 12 for note in TARGET_NOTES]
assert len(LYRICS) == len(DURATIONS) == len(TARGET_NOTES) == 25


def main() -> None:
    if not (MODEL / "Checkpoint" / "S_200000.pt").exists():
        raise SystemExit("Model missing. Run scripts/fetch_diffsingerkr.py first.")
    sys.path.insert(0, str(MODEL))
    os.chdir(MODEL)
    from Inference import Inferencer  # noqa: E402

    started = time.perf_counter()
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    free_before = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else None
    print(f"PyTorch {torch.__version__}; device: {device}", flush=True)
    inferencer = Inferencer(
        hp_path="Hyper_Parameters.yaml",
        checkpoint_path="Checkpoint/S_200000.pt",
        batch_size=1,
    )
    samples = inferencer.Inference_Epoch(
        message_times_list=[DURATIONS],
        lyrics=[LYRICS],
        notes=[MODEL_NOTES],
        singers=["CSD"],
        genres=["Children"],
        ddim_steps=20,
        use_tqdm=False,
    )[0]
    samples = np.asarray(samples, dtype=np.float32)
    if not np.isfinite(samples).all() or samples.size == 0:
        raise RuntimeError("Inference returned empty or non-finite audio")
    # The bundled TorchScript vocoder returns PCM values on an int16 scale.
    raw_peak = float(np.max(np.abs(samples)))
    if raw_peak > 1.5:
        samples /= 32768.0
    if float(np.max(np.abs(samples))) > 1.0:
        raise RuntimeError("Vocoder produced samples outside the PCM range")
    sample_rate = int(inferencer.hp.Sound.Sample_Rate)
    wav_path = OUT / "proof_korean_svs.wav"
    sf.write(wav_path, samples, sample_rate, subtype="PCM_16")
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    result = {
        "status": "rendered",
        "model": "CODEJIN/DiffSingerKR",
        "input_lyrics": "".join(LYRICS),
        "input_syllables": LYRICS,
        "target_notes_midi": TARGET_NOTES,
        "model_note_ids": MODEL_NOTES,
        "model_note_offset_semitones": 12,
        "input_durations_seconds": DURATIONS,
        "expected_seconds": sum(DURATIONS),
        "actual_seconds": len(samples) / sample_rate,
        "sample_rate": sample_rate,
        "rms": rms,
        "peak": float(np.max(np.abs(samples))),
        "raw_vocoder_peak": raw_peak,
        "device": device,
        "torch": torch.__version__,
        "render_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_mb": (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
            if torch.cuda.is_available() else None
        ),
        "peak_cuda_reserved_mb": (
            torch.cuda.max_memory_reserved() / (1024 * 1024)
            if torch.cuda.is_available() else None
        ),
        "cuda_free_memory_delta_mb": (
            (free_before - torch.cuda.mem_get_info()[0]) / (1024 * 1024)
            if free_before is not None else None
        ),
        "pronunciation_review": "pending human listening",
        "pitch_review": "pending F0 measurement",
    }
    (OUT / "proof_korean_svs.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if rms < 0.0001:
        raise RuntimeError("Rendered audio is nearly silent")


if __name__ == "__main__":
    main()
