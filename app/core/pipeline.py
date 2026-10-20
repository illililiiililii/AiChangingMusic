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
    if (not project.analysis.get("notes") and not project.beat_offset_seconds
            and result.get("beat_times")):
        project.beat_offset_seconds = result["beat_times"][0]
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


def extract_basic_pitch(project: Project, progress: Callable[[str], None] | None = None) -> str:
    """Produce a reviewable MIDI candidate; never replace hand-edited notes."""
    vocal = project.artifact("vocals.wav")
    if not vocal.is_file():
        raise ValueError("보컬 분리를 먼저 실행하세요.")
    executable = ROOT / ".venv-basicpitch" / "Scripts" / "basic-pitch.exe"
    if not executable.is_file():
        raise FileNotFoundError("Basic Pitch가 없습니다. scripts/setup_basic_pitch.ps1을 실행하세요.")
    output = project.artifact("basic_pitch")
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    _run([str(executable), str(output), str(vocal)], env=environment, progress=progress)
    midi_files = sorted(output.glob("*.mid"), key=lambda path: path.stat().st_mtime)
    if not midi_files:
        raise RuntimeError("Basic Pitch가 MIDI 파일을 만들지 못했습니다. logs 폴더를 확인하세요.")
    from app.music_analysis.midi_io import read_midi
    notes, midi_bpm = read_midi(midi_files[-1])
    if not notes:
        raise ValueError("MIDI에 음표가 없습니다. 분리된 보컬을 확인하세요.")
    project.analysis["basic_pitch_notes"] = notes
    project.analysis["basic_pitch_midi"] = str(midi_files[-1])
    project.analysis["basic_pitch_bpm"] = midi_bpm
    project.save()
    return f"Basic Pitch 음표 후보 {len(notes)}개 생성 · 피아노롤에서 검토 후 적용하세요"


def install_basic_pitch(project: Project, progress: Callable[[str], None] | None = None) -> str:
    if progress:
        progress("Basic Pitch 선택 설치 중 — Python 3.10 전용 환경을 만듭니다")
    _run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
          str(ROOT / "scripts" / "setup_basic_pitch.ps1")], cwd=ROOT, progress=progress)
    return "Basic Pitch 설치 완료"


def align_lyrics(project: Project) -> str:
    from app.lyric_generation.alignment import align
    notes = project.analysis.get("notes")
    if not notes:
        raise ValueError("멜로디 분석을 먼저 실행하세요.")
    if not project.parody_lyrics.strip():
        raise ValueError("개사 가사를 입력하세요.")
    result = align(project.parody_lyrics, notes, project.original_lyrics)
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


def sing(project: Project, progress: Callable[[str], None] | None = None) -> str:
    score = project.artifact("score.json")
    if not score.is_file():
        raise ValueError("가사 정렬을 먼저 실행하세요.")
    result = project.artifact("singing.wav")
    if project.singing_engine == "korean_tts":
        _run([str(RVC_PYTHON), str(ROOT / "scripts" / "render_korean_singing.py"),
              str(project.path), str(result)], progress=progress)
    elif project.singing_engine == "diffsinger":
        _run([sys.executable, str(ROOT / "scripts" / "render_singing.py"),
              str(score), str(result), "--gpu-index", "0"])
    else:
        raise ValueError("노래 합성 방식을 다시 선택하세요.")
    events = json.loads(score.read_text(encoding="utf-8"))["syllables"]
    short = sum(event["end"] - event["start"] < .15 for event in events)
    suffix = f" · 0.15초 미만 {short}음절은 발음을 확인하세요" if short else ""
    return f"노래 보컬 생성: {result.name}{suffix}"


def sing_segment(project: Project, progress: Callable[[str], None] | None = None) -> str:
    score = project.artifact("score.json")
    output = project.artifact("singing.wav")
    if not score.is_file() or not output.is_file():
        raise ValueError("전체 노래 합성을 먼저 실행하세요.")
    if project.segment_end <= project.segment_start:
        raise ValueError("재생성할 구간을 올바르게 설정하세요.")
    if project.singing_engine == "korean_tts":
        result = sing(project, progress)
        project.stages.pop("convert", None)
        project.stages.pop("mix", None)
        project.save()
        return result + " · 발음 우선 방식은 전체 보컬을 갱신했습니다"
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


def convert(project: Project, progress: Callable[[str], None] | None = None) -> str:
    if not project.voice_consent:
        raise PermissionError("음성 학습·변환 허락 확인이 필요합니다.")
    model, index = Path(project.voice_model), Path(project.voice_index)
    if not model.is_file():
        raise FileNotFoundError("RVC 모델을 선택하세요.")
    if project.conversion_index_rate > 0 and not index.is_file():
        raise FileNotFoundError("RVC 인덱스를 선택하거나 음색 반영 강도를 0으로 설정하세요.")
    if project.conversion_source in ("recording", "openutau"):
        if not project.guide_vocal:
            raise ValueError("직접 녹음한 보컬 파일을 선택하세요.")
        source = decode(Path(project.guide_vocal), project.artifact("guide_vocal.wav"), channels=1)
        if project.guide_offset_seconds:
            audio, rate = sf.read(source, dtype="float32")
            shift = round(project.guide_offset_seconds * rate)
            if shift < 0 and -shift >= len(audio):
                raise ValueError("녹음 시작 위치 조정이 녹음 길이보다 큽니다.")
            audio = np.pad(audio, (shift, 0)) if shift > 0 else audio[-shift:] if shift < 0 else audio
            sf.write(source, audio, rate)
    elif project.conversion_source == "synth":
        source = project.artifact("singing.wav")
        if not source.is_file():
            raise ValueError("노래 합성을 먼저 실행하세요.")
        if project.stages.get("sing", {}).get("status") != "완료":
            raise ValueError("노래 합성 방식이나 가사가 변경되었습니다. 노래 합성을 다시 실행하세요.")
    else:
        raise ValueError("음색 변환 입력을 다시 선택하세요.")
    if not RVC_PYTHON.is_file():
        raise FileNotFoundError(".venv-rvc가 없습니다.")
    from app.core.rvc_conversion import convert_audio
    return convert_audio(source, project.artifact("converted.wav"), python=RVC_PYTHON,
                         model=model, index=index if project.conversion_index_rate > 0 else None,
                         pitch=project.conversion_pitch_semitones,
                         index_rate=project.conversion_index_rate,
                         protect=project.conversion_protect,
                         gpu_index=_gpu_with_free_memory(), progress=progress)


def mix(project: Project) -> str:
    from app.mixing.render import render_mix
    vocal = project.artifact("converted.wav")
    instrumental = project.artifact("instrumental.wav")
    if not vocal.is_file() or not instrumental.is_file():
        raise ValueError("음색 변환 및 반주 분리 결과가 필요합니다.")
    if project.stages.get("convert", {}).get("status") != "완료":
        raise ValueError("음색 변환을 완료한 뒤 믹싱을 실행하세요.")
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
            "analyze": analyze, "install_basic_pitch": install_basic_pitch,
            "basic_pitch": extract_basic_pitch,
            "generate": generate, "lyrics": align_lyrics, "train": train,
            "sing": sing, "sing_segment": sing_segment,
            "convert": convert, "mix": mix}


def run_stage(path: Path, stage: str, progress: Callable[[str], None] | None = None) -> str:
    if stage not in HANDLERS:
        raise ValueError(stage)
    project = Project.load(path)
    record(stage, "시작", f"프로젝트: {path}")
    project.mark(stage, "진행 중")
    try:
        result = HANDLERS[stage](project, progress) if stage in ("install_basic_pitch", "basic_pitch", "train", "sing", "sing_segment", "convert") else HANDLERS[stage](project)
        if stage == "sing":
            for dependent in ("convert", "mix"):
                project.stages.pop(dependent, None)
        elif stage == "convert":
            project.stages.pop("mix", None)
        project.mark(stage, "완료", result)
        record(stage, "완료", result)
        return result
    except Exception as exc:
        project.mark(stage, "실패", str(exc)[:250])
        record(stage, "실패", traceback.format_exc())
        raise
