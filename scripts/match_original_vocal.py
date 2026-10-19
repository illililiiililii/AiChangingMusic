"""Transfer the original separated vocal's pitch contour to a guide vocal."""

from __future__ import annotations

import sys
from pathlib import Path

import librosa
import numpy as np
import pyworld
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
RVC = ROOT / "external" / "RVC"


def original_f0(path: Path) -> np.ndarray:
    """RMVPE produces one F0 value per 10 ms of the original recording."""
    sys.path.insert(0, str(RVC))
    from infer.rmvpe import RMVPE

    audio, _ = librosa.load(str(path), sr=16000, mono=True)
    model = RMVPE(str(RVC / "assets" / "rmvpe" / "rmvpe.pt"), False, "cuda")
    chunks = []
    for start in range(0, len(audio), 16000 * 15):
        chunks.append(model.infer_from_audio(audio[start:start + 16000 * 15]))
    return np.concatenate(chunks)


def follow_pitch(guide: np.ndarray, reference_f0: np.ndarray, sample_rate: int) -> np.ndarray:
    """Keep guide formants/articulation while using the original sung F0."""
    output = np.zeros_like(guide, dtype=np.float32)
    chunk_samples = sample_rate * 8
    for begin in range(0, len(guide), chunk_samples):
        segment = guide[begin:begin + chunk_samples].astype(np.float64)
        if np.max(np.abs(segment)) < .001:
            continue
        source_f0, time_axis = pyworld.dio(segment, sample_rate, frame_period=10.)
        source_f0 = pyworld.stonemask(segment, source_f0, time_axis, sample_rate)
        spectrum = pyworld.cheaptrick(segment, source_f0, time_axis, sample_rate)
        aperiodicity = pyworld.d4c(segment, source_f0, time_axis, sample_rate)
        absolute_frames = begin / sample_rate * 100 + np.arange(len(source_f0))
        reference = np.interp(absolute_frames, np.arange(len(reference_f0)), reference_f0,
                              left=0., right=0.)
        target_f0 = np.where((source_f0 > 0) & (reference > 0), reference, source_f0)
        synthesized = pyworld.synthesize(target_f0, spectrum, aperiodicity, sample_rate,
                                          frame_period=10.)
        length = min(len(segment), len(synthesized))
        output[begin:begin + length] = synthesized[:length]
    return output


def render(guide_path: Path, original_path: Path, output_path: Path) -> None:
    guide, sr = sf.read(guide_path, dtype="float32")
    if guide.ndim > 1:
        guide = guide.mean(axis=1)
    f0 = original_f0(original_path)
    shifted = follow_pitch(guide, f0, sr)
    peak = np.max(np.abs(shifted))
    if peak > .98:
        shifted *= .98 / peak
    sf.write(output_path, shifted, sr, subtype="PCM_16")


if __name__ == "__main__":
    render(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
