"""Independent, resumable local audio stages used by the GUI and CLI."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from uuid import uuid4
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from app.audio_processing.io import decode, ffmpeg
from app.core.project import Project
from app.core.activity_log import record
from app.core.settings import ROOT

RVC_PYTHON = ROOT / ".venv-rvc" / "Scripts" / "python.exe"


def _run(command: list[str], *, cwd: Path | None = None, env: dict | None = None,
         progress: Callable[[str], None] | None = None) -> None:
    if progress:
        lines = []
        with subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1) as process:
            assert process.stdout is not None
            for line in process.stdout:
                lines.append(line)
                if line.strip() and ("===" in line or "Prepared" in line or "MODEL=" in line or "INDEX=" in line):
                    progress(line.strip())
            code = process.wait()
        if code:
            raise RuntimeError(f"명령 실패 ({code}): {' '.join(command[:5])}\n{''.join(lines)[-3000:]}")
        return
    result = subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, errors="replace")
    if result.returncode:
        raise RuntimeError(f"명령 실패 ({result.returncode}): {' '.join(command[:5])}\n{result.stdout[-3000:]}\n{result.stderr[-3000:]}")


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _gpu_with_free_memory(minimum_mb: int = 3300) -> str:
    smi = Path("C:/Windows/System32/nvidia-smi.exe")
    if not smi.is_file():
        return ""
    try:
        result = subprocess.run([str(smi), "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, check=True, timeout=5)
        devices = [tuple(int(part.strip()) for part in line.split(",")[:2])
                   for line in result.stdout.splitlines() if "," in line]
        devices = [(index, free) for index, free in devices if free >= minimum_mb]
        return str(max(devices, key=lambda item: item[1])[0]) if devices else ""
    except (OSError, ValueError, subprocess.SubprocessError):
        return ""


def import_audio(project: Project) -> str:
    if not project.source:
        raise ValueError("원곡 파일을 먼저 선택하세요.")
    path = decode(Path(project.source), project.artifact("source.wav"))
    return f"가져옴: {path.name}"


def separate(project: Project) -> str:
    source = project.artifact("source.wav")
    if not source.is_file():
        raise ValueError("원곡 가져오기를 먼저 실행하세요.")
    if not RVC_PYTHON.is_file():
        raise FileNotFoundError(".venv-rvc가 없습니다. setup.ps1을 실행하세요.")
    out = project.artifact("demucs")
    env = os.environ.copy()
    env["PATH"] = str(ffmpeg().parent) + os.pathsep + env.get("PATH", "")
    env["TORCH_HOME"] = str(ROOT / "models" / "torch")
    # Demucs weights must be installed beforehand; this stage never downloads silently.
    env["HF_HUB_OFFLINE"] = "1"
    command = [str(RVC_PYTHON), "-m", "demucs", "--two-stems", "vocals", "-n", "htdemucs",
               "--segment", "7", "--overlap", "0.1", "-d", "cuda:0", "-o", str(out), str(source)]
    try:
        _run(command, env=env)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() and "cuda" not in str(exc).lower():
            raise
        command[command.index("cuda:0")] = "cpu"
        _run(command, env=env)
    stems = out / "htdemucs" / "source"
    decode(stems / "vocals.wav", project.artifact("vocals.wav"), channels=1)
    decode(stems / "no_vocals.wav", project.artifact("instrumental.wav"), channels=2)
    return "보컬·반주 분리 완료"


def transcribe(project: Project) -> str:
    vocal = project.artifact("vocals.wav")
    if not vocal.is_file():
        raise ValueError("보컬 분리를 먼저 실행하세요.")
    model = ROOT / "models" / "faster-whisper-base"
    if not model.is_dir():
        raise FileNotFoundError("Whisper 모델이 없습니다. scripts/download_models.py를 실행하세요.")
    if not RVC_PYTHON.is_file():
        raise FileNotFoundError(".venv-rvc가 없습니다.")
    output = project.artifact("transcription.json")
    previous_recognition = ""
    if output.is_file():
        previous_data = json.loads(output.read_text(encoding="utf-8"))
        previous_recognition = "\n".join(segment.get("text", "") for segment in previous_data.get("segments", []))
    keep_edited_lyrics = bool(project.original_lyrics.strip()
                              and project.original_lyrics.strip() != previous_recognition.strip())
    _run([str(RVC_PYTHON), str(ROOT / "scripts" / "transcribe_local.py"), str(vocal),
          str(model), str(output), "--language", project.source_language])
    data = json.loads(output.read_text(encoding="utf-8"))
    if not keep_edited_lyrics:
        project.original_lyrics = "\n".join(segment["text"] for segment in data["segments"])
    project.save()
    suffix = ", 직접 수정한 가사는 유지" if keep_edited_lyrics else " (수정 권장)"
    return f"가사 인식: {len(data['segments'])}개 구간{suffix}"


def analyze(project: Project) -> str:
    vocal = project.artifact("vocals.wav")
    if not vocal.is_file():
        raise ValueError("보컬 분리를 먼저 실행하세요.")
    from app.music_analysis.melody import analyze_vocal
    from app.music_analysis.syllables import estimate_syllables
    result = analyze_vocal(vocal)
    transcription = project.artifact("transcription.json")
    if transcription.is_file():
        transcript = json.loads(transcription.read_text(encoding="utf-8"))
        result["original_syllables"] = estimate_syllables(
            transcript, result["notes"], project.original_lyrics)
        recognized = "\n".join(segment.get("text", "") for segment in transcript.get("segments", []))
        edited = bool(project.original_lyrics.strip()
                      and project.original_lyrics.strip() != recognized.strip())
        result["syllable_timing_method"] = (
            "수정한 원곡 가사를 Whisper 단어 시간에 분배한 추정; 수동 검토 필요" if edited
            else "Whisper 단어 구간의 균등 분할 추정; 수동 검토 필요")
    project.analysis.update(result)
    project.save()
    _json(project.artifact("analysis.json"), result)
    return f"BPM {result['bpm']:.1f}, 음표 {len(result['notes'])}개 추정"


def align_lyrics(project: Project) -> str:
    from app.lyric_generation.alignment import align
    notes = project.analysis.get("notes")
    if not notes:
        raise ValueError("멜로디 분석을 먼저 실행하세요.")
    if not project.parody_lyrics.strip():
        raise ValueError("개사 가사를 입력하세요.")
    result = align(project.parody_lyrics, notes)
    project.analysis["lyric_fit"] = (
        f"부분 {len(result['syllables'])}/{len(notes)}음표 · 음절 적합도 {result['score']:.0f}/100"
        if result.get("partial") else f"{result['score']:.0f}/100")
    project.analysis["alignment"] = result["syllables"]
    project.save()
    _json(project.artifact("score.json"), result)
    if result.get("partial"):
        return (f"부분 가사 {len(result['syllables'])}/{len(notes)}개 음표에 정렬 완료 "
                "(나머지 구간에는 새 보컬이 없습니다)")
    return f"가사 {len(result['syllables'])}개 음절 정렬, 적합도 {result['score']:.0f}/100"


def generate(project: Project) -> str:
    from app.lyric_generation.ollama import generate_lyrics
    notes = project.analysis.get("notes", [])
    if not notes:
        raise ValueError("원곡 멜로디를 먼저 분석하세요.")
    text = generate_lyrics(project.theme, project.lyric_style, project.original_lyrics,
                           notes, project.llm_model)
    project.parody_lyrics = text
    project.save()
    from app.lyric_generation.alignment import syllables
    count = len(syllables(text))
    return (f"로컬 AI 개사 생성: {count}/{len(notes)}음절" +
            (" (음표 수에 맞게 수정 후 정렬하세요)" if count != len(notes) else " (검토 후 정렬하세요)"))


def sing(project: Project) -> str:
    score = project.artifact("score.json")
    if not score.is_file():
        raise ValueError("가사 정렬을 먼저 실행하세요.")
    result = project.artifact("singing.wav")
    _run([sys.executable, str(ROOT / "scripts" / "render_singing.py"), str(score), str(result), "--gpu-index", "0"])
    events = json.loads(score.read_text(encoding="utf-8"))["syllables"]
    short = sum(event["end"] - event["start"] < .15 for event in events)
    suffix = f" · 0.15초 미만 {short}음절은 발음을 확인하세요" if short else ""
    return f"노래 보컬 생성: {result.name}{suffix}"


def sing_segment(project: Project) -> str:
    score = project.artifact("score.json")
    output = project.artifact("singing.wav")
    if not score.is_file() or not output.is_file():
        raise ValueError("전체 노래 합성을 먼저 실행하세요.")
    if project.segment_end <= project.segment_start:
        raise ValueError("재생성할 구간을 올바르게 설정하세요.")
    _run([sys.executable, str(ROOT / "scripts" / "render_singing.py"), str(score), str(output),
          "--gpu-index", "0", "--start", str(project.segment_start), "--end", str(project.segment_end)])
    project.stages.pop("convert", None)
    project.stages.pop("mix", None)
    project.save()
    return f"{project.segment_start:.2f}–{project.segment_end:.2f}초 노래 재생성 완료"


def train(project: Project, progress: Callable[[str], None] | None = None) -> str:
    if not project.voice_consent:
        raise PermissionError("음성 학습·변환 허락 확인이 필요합니다.")
    if not project.voice_source:
        raise ValueError("말하기 음성 파일 또는 폴더를 선택하세요.")
    name = "profile_" + project.directory.name.encode("utf-8").hex()[:16] + "_" + uuid4().hex[:8]
    _run([str(RVC_PYTHON), str(ROOT / "scripts" / "train_voice_profile.py"),
          project.voice_source, name, "--authorized"], progress=progress)
    models = sorted((ROOT / "external" / "RVC" / "assets" / "weights").glob(f"{name}_e*.pth"),
                    key=lambda p: p.stat().st_mtime)
    indices = sorted((ROOT / "external" / "RVC" / "assets" / "indices").glob(f"*{name}*.index"),
                     key=lambda p: p.stat().st_mtime)
    project.voice_model, project.voice_index = str(models[-1]), str(indices[-1])
    project.save()
    return f"음성 프로필 학습 완료: {name}"


def convert(project: Project) -> str:
    if not project.voice_consent:
        raise PermissionError("음성 학습·변환 허락 확인이 필요합니다.")
    model, index = Path(project.voice_model), Path(project.voice_index)
    if not model.is_file():
        raise FileNotFoundError("RVC 모델을 선택하세요.")
    if project.conversion_index_rate > 0 and not index.is_file():
        raise FileNotFoundError("RVC 인덱스를 선택하거나 음색 반영 강도를 0으로 설정하세요.")
    if project.conversion_source == "recording":
        if not project.guide_vocal:
            raise ValueError("직접 녹음한 보컬 파일을 선택하세요.")
        source = decode(Path(project.guide_vocal), project.artifact("guide_vocal.wav"), channels=1)
    elif project.conversion_source == "synth":
        source = project.artifact("singing.wav")
        if not source.is_file():
            raise ValueError("노래 합성을 먼저 실행하세요.")
    else:
        raise ValueError("음색 변환 입력을 다시 선택하세요.")
    if not RVC_PYTHON.is_file():
        raise FileNotFoundError(".venv-rvc가 없습니다.")
    target = project.artifact("converted.wav")
    env = os.environ.copy()
    env.update({"PATH": str(ffmpeg().parent) + os.pathsep + env.get("PATH", ""),
                "CUDA_VISIBLE_DEVICES": _gpu_with_free_memory(), "HF_HUB_OFFLINE": "1"})
    command = [str(RVC_PYTHON), "-m", "infer.cli", "--model", str(model), "--input", str(source),
               "--output", str(target), "--pitch", str(project.conversion_pitch_semitones),
               "--f0-method", "rmvpe", "--protect", str(project.conversion_protect),
               "--index-rate", str(project.conversion_index_rate), "--overwrite"]
    if project.conversion_index_rate > 0:
        command.extend(["--index", str(index)])
    _run(command, cwd=ROOT / "external" / "RVC", env=env)
    return f"음색 변환: {target.name}"


def mix(project: Project) -> str:
    from app.mixing.render import render_mix
    vocal = project.artifact("converted.wav")
    instrumental = project.artifact("instrumental.wav")
    if not vocal.is_file() or not instrumental.is_file():
        raise ValueError("음색 변환 및 반주 분리 결과가 필요합니다.")
    wav = project.artifact("final.wav")
    render_mix(vocal, instrumental, wav, project.vocal_gain, project.instrumental_gain,
               reverb=project.reverb)
    mp3 = project.artifact("final.mp3")
    _run([str(ffmpeg()), "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav),
          "-codec:a", "libmp3lame", "-b:a", "256k", str(mp3)])
    _json(project.artifact("final.json"), {"ai_generated": True, "source": project.source,
        "voice_model": project.voice_model, "conversion_source": project.conversion_source,
        "guide_vocal": project.guide_vocal if project.conversion_source == "recording" else "",
        "wav": str(wav), "mp3": str(mp3)})
    return f"완료: {wav.name}, {mp3.name} · AI 생성 음성"


HANDLERS = {"import": import_audio, "separate": separate, "transcribe": transcribe,
            "analyze": analyze, "generate": generate, "lyrics": align_lyrics, "train": train,
            "sing": sing, "sing_segment": sing_segment,
            "convert": convert, "mix": mix}


def run_stage(path: Path, stage: str, progress: Callable[[str], None] | None = None) -> str:
    if stage not in HANDLERS:
        raise ValueError(stage)
    project = Project.load(path)
    record(stage, "시작", f"프로젝트: {path}")
    project.mark(stage, "진행 중")
    try:
        result = train(project, progress) if stage == "train" else HANDLERS[stage](project)
        project.mark(stage, "완료", result)
        record(stage, "완료", result)
        return result
    except Exception as exc:
        project.mark(stage, "실패", str(exc)[:250])
        record(stage, "실패", traceback.format_exc())
        raise
