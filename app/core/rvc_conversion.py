"""Chunked RVC conversion, diagnostic streaming, and sample-aligned assembly."""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable
from uuid import uuid4

import numpy as np
import soundfile as sf

from app.core.activity_log import LOG_DIR, record
from app.core.settings import ROOT

SEGMENT_SECONDS = 15
OVERLAP_SECONDS = .3
STALL_SECONDS = 180
EVENT_PREFIX = "RVC_EVENT "
PROGRESS_PREFIX = "RVC_PROGRESS "
CUDA_ERROR = re.compile(r"cuda.*(?:out of memory|error)|(?:out of memory).*cuda|cublas.*error", re.I)


def _gpu_snapshot() -> str:
    executable = Path("C:/Windows/System32/nvidia-smi.exe")
    if not executable.is_file():
        return "nvidia-smi 없음"
    try:
        result = subprocess.run(
            [str(executable), "--query-gpu=index,name,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        return result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)


def _progress(callback: Callable[[str], None] | None, **values) -> None:
    if callback:
        callback(PROGRESS_PREFIX + json.dumps(values, ensure_ascii=False))


def prepare_chunks(source: Path, input_dir: Path, *, seconds: int = SEGMENT_SECONDS,
                   overlap: float = OVERLAP_SECONDS) -> tuple[list[dict], int, int]:
    info = sf.info(source)
    if info.frames <= 0:
        raise ValueError("변환 입력 오디오가 비어 있습니다.")
    if info.samplerate <= 0:
        raise ValueError("변환 입력의 샘플레이트가 올바르지 않습니다.")
    core_frames = max(1, round(seconds * info.samplerate))
    overlap_frames = round(overlap * info.samplerate)
    plans = []
    input_dir.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(source) as audio:
        boundaries = [0]
        for ideal in range(core_frames, info.frames, core_frames):
            radius = round(.6 * info.samplerate)
            left, right = max(boundaries[-1] + 1, ideal - radius), min(info.frames - 1, ideal + radius)
            audio.seek(left)
            candidates = audio.read(right - left, dtype="float32", always_2d=True).mean(axis=1)
            envelope = np.abs(candidates)
            window = max(1, round(.02 * info.samplerate))
            count = len(envelope) // window
            if count:
                energy = np.mean(envelope[:count * window].reshape(count, window), axis=1)
                boundary = left + int(np.argmin(energy)) * window + window // 2
            else:
                boundary = ideal
            boundaries.append(min(max(boundary, boundaries[-1] + 1), info.frames - 1))
        boundaries.append(info.frames)
        for number, (core_start, core_end) in enumerate(zip(boundaries, boundaries[1:]), 1):
            extended_start = max(0, core_start - overlap_frames)
            extended_end = min(info.frames, core_end + overlap_frames)
            audio.seek(extended_start)
            samples = audio.read(extended_end - extended_start, dtype="float32", always_2d=True)
            mono = samples.mean(axis=1)
            name = f"{number:04d}.wav"
            sf.write(input_dir / name, mono, info.samplerate, subtype="PCM_16")
            plans.append({"name": name, "core_start": core_start, "core_end": core_end,
                          "start": extended_start, "end": extended_end})
    return plans, info.samplerate, info.frames


def merge_chunks(plans: list[dict], output_dir: Path, source_rate: int,
                 source_frames: int, target: Path) -> None:
    if not plans:
        raise ValueError("병합할 변환 구간이 없습니다.")
    output_rate = None
    mixed = weights = None
    for index, plan in enumerate(plans):
        path = output_dir / plan["name"]
        if not path.is_file() or path.stat().st_size <= 44:
            raise RuntimeError(f"변환 구간 {index + 1}/{len(plans)} 출력이 누락됐습니다: {path}")
        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        if audio.shape[1] != 1:
            raise ValueError(f"변환 구간 {index + 1}이 모노가 아닙니다.")
        if output_rate is None:
            output_rate = rate
            total = round(source_frames * rate / source_rate)
            mixed = np.zeros(total, dtype="float32")
            weights = np.zeros(total, dtype="float32")
        elif rate != output_rate:
            raise ValueError(f"변환 구간 {index + 1}의 샘플레이트가 다릅니다: {rate}")

        start = round(plan["start"] * rate / source_rate)
        end = round(plan["end"] * rate / source_rate)
        expected = end - start
        if abs(len(audio) - expected) > round(rate * .25):
            raise RuntimeError(f"변환 구간 {index + 1}의 길이가 원본과 {abs(len(audio)-expected)/rate:.2f}초 다릅니다.")
        chunk = np.zeros(expected, dtype="float32")
        chunk[:min(expected, len(audio))] = audio[:expected, 0]
        fade = np.ones(expected, dtype="float32")
        if index:
            boundary = round(plan["core_start"] * rate / source_rate)
            width = min(expected, max(1, 2 * (boundary - start)))
            fade[:width] = np.linspace(0, 1, width, endpoint=True, dtype="float32")
        if index + 1 < len(plans):
            boundary = round(plan["core_end"] * rate / source_rate)
            width = min(expected, max(1, 2 * (end - boundary)))
            fade[-width:] = np.minimum(fade[-width:],
                                        np.linspace(1, 0, width, endpoint=True, dtype="float32"))
        mixed[start:end] += chunk * fade
        weights[start:end] += fade
    if np.any(weights <= 0):
        raise RuntimeError("구간 병합 후 오디오에 누락된 샘플이 있습니다.")
    mixed /= weights
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    try:
        sf.write(temporary, mixed, output_rate, subtype="PCM_16", format="WAV")
        rendered = sf.info(temporary)
        if abs(rendered.duration - source_frames / source_rate) > .05:
            raise RuntimeError("최종 변환 보컬 길이가 입력 오디오와 맞지 않습니다.")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _terminate_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                       capture_output=True, timeout=10)
    else:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def _cancel_requested(env: dict[str, str]) -> bool:
    path = env.get("AICHANGINGMUSIC_CANCEL_FILE")
    return bool(path and Path(path).is_file())


def _run_batch(command: list[str], *, env: dict[str, str], total: int,
               progress: Callable[[str], None] | None) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logfile = LOG_DIR / f"rvc-{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:8]}.log"
    record("convert", "RVC 프로세스 준비", f"로그: {logfile}\n명령: {subprocess.list2cmdline(command)}")
    events: queue.Queue[tuple[str, str | None]] = queue.Queue()
    start_time = time.monotonic()
    completed = 0
    current = 0
    durations: list[float] = []
    current_started = None
    last_change = start_time
    last_warning = 0.0
    warning_text = ""
    last_emit = start_time
    gpu = f"CUDA:0 (GPU {env['CUDA_VISIBLE_DEVICES']})" if env.get("CUDA_VISIBLE_DEVICES") else "CPU 또는 RVC 자동 선택"
    issue = ""
    recent: list[str] = []
    _progress(progress, total=total, current=0, completed=0, percent=0.,
              elapsed=0, eta=None, gpu=gpu, status="RVC 시작 중", warning="")
    with logfile.open("a", encoding="utf-8", buffering=1) as log:
        log.write(f"{datetime.now().astimezone().isoformat()} START\n")
        log.write(f"command={subprocess.list2cmdline(command)}\n")
        log.write(f"gpu={gpu}; model={command[command.index('--model') + 1]}; "
                  f"index_rate={command[command.index('--index-rate') + 1]}; "
                  f"pitch={command[command.index('--pitch') + 1]}; "
                  f"protect={command[command.index('--protect') + 1]}; f0=rmvpe; chunks={total}\n")
        log.write(f"gpu_snapshot={_gpu_snapshot()}\n")
        with subprocess.Popen(command, cwd=ROOT / "external" / "RVC", env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                              text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
            log.write(f"pid={process.pid}\n")
            record("convert", "RVC 프로세스 시작", f"PID {process.pid}; GPU {gpu}; 구간 {total}개")

            def read_stream(kind: str, stream):
                try:
                    for line in stream:
                        events.put((kind, line.rstrip("\r\n")))
                finally:
                    events.put((kind, None))

            readers = [threading.Thread(target=read_stream, args=(kind, stream), daemon=True)
                       for kind, stream in (("stdout", process.stdout), ("stderr", process.stderr))]
            for reader in readers:
                reader.start()
            closed = 0
            while closed < 2 or process.poll() is None:
                if _cancel_requested(env):
                    log.write(f"{datetime.now().astimezone().isoformat()} CANCEL requested\n")
                    _terminate_tree(process)
                    record("convert", "사용자 중단", f"PID {process.pid}; 완료 {completed}/{total}")
                    raise InterruptedError("사용자가 RVC 변환을 중단했습니다.")
                try:
                    kind, line = events.get(timeout=1)
                except queue.Empty:
                    kind, line = "", ""
                now = time.monotonic()
                changed = False
                if line is None:
                    closed += 1
                elif line:
                    log.write(f"{datetime.now().astimezone().isoformat()} [{kind}] {line}\n")
                    recent.append(f"[{kind}] {line}")
                    recent = recent[-40:]
                    if CUDA_ERROR.search(line):
                        issue = line
                        warning_text = f"[WARNING] CUDA 오류 감지: {line}"
                        record("convert", "CUDA 경고", warning_text)
                        _progress(progress, total=total, current=current, completed=completed,
                                  percent=100 * completed / total, elapsed=int(now-start_time),
                                  eta=None, gpu=gpu, status="CUDA 오류 감지", warning=warning_text)
                    if kind == "stdout" and line.startswith(EVENT_PREFIX):
                        try:
                            data = json.loads(line[len(EVENT_PREFIX):])
                        except ValueError:
                            data = {}
                        if data.get("type") == "ready":
                            gpu = f"{data.get('device')} (GPU {env.get('CUDA_VISIBLE_DEVICES', '자동')})"
                            log.write(f"device={gpu}; cuda_visible={data.get('cuda_visible_devices')}\n")
                        elif data.get("type") == "start":
                            current = int(data["segment"])
                            current_started = now
                            changed = True
                            record("convert", "구간 시작", f"{current}/{total}; 입력 {data.get('input')}")
                        elif data.get("type") == "done":
                            number = int(data["segment"])
                            if number != completed + 1:
                                raise RuntimeError(f"RVC 구간 완료 순서가 올바르지 않습니다: {number}")
                            completed = number
                            elapsed_segment = now - current_started if current_started else 0.
                            durations.append(elapsed_segment)
                            last_change = now
                            last_warning = 0.
                            warning_text = ""
                            changed = True
                            record("convert", "구간 완료", f"{completed}/{total}; {elapsed_segment:.1f}초; 출력 {data.get('output')}")
                expected_overrun = bool(durations and current_started and
                                        now - current_started > max(90., 3 * sum(durations) / len(durations)))
                stalled = now - last_change >= STALL_SECONDS
                if (completed < total and process.poll() is None and
                        (stalled or expected_overrun) and now - last_warning >= 60):
                    last_warning = now
                    warning_text = (f"[WARNING] RVC 변환 지연 감지 · 현재 {current}/{total} · "
                                    f"마지막 완료 {int(now-last_change)}초 전 · 프로세스 실행 중 · GPU {gpu} · "
                                    "작업이 정상적으로 진행 중인지 확인이 필요합니다.")
                    log.write(f"{datetime.now().astimezone().isoformat()} {warning_text}\n")
                    log.write(f"gpu_snapshot={_gpu_snapshot()}\n")
                    record("convert", "지연 경고", warning_text)
                    changed = True
                eta = round(sum(durations) / len(durations) * (total-completed)) if durations else None
                if changed or now - last_emit >= 1:
                    _progress(progress, total=total, current=current, completed=completed,
                              percent=round(100 * completed / total, 1), elapsed=int(now-start_time),
                              eta=eta, gpu=gpu, status="RVC 음색 변환 중" if current else "모델 준비 중",
                              warning=warning_text)
                    last_emit = now
            code = process.wait()
            log.write(f"{datetime.now().astimezone().isoformat()} EXIT code={code}; completed={completed}/{total}\n")
            record("convert", "RVC 프로세스 종료", f"PID {process.pid}; 코드 {code}; 완료 {completed}/{total}; 로그 {logfile}")
    if code:
        cause = f"CUDA 오류: {issue}" if issue else "\n".join(recent[-12:])
        raise RuntimeError(f"RVC 비정상 종료 (코드 {code}, {completed}/{total}구간). {cause}\n로그: {logfile}")
    if completed != total:
        raise RuntimeError(f"RVC 출력 구간 누락: {completed}/{total}. 로그: {logfile}")


def convert_audio(source: Path, target: Path, *, python: Path, model: Path,
                  index: Path | None, pitch: int, index_rate: float, protect: float,
                  gpu_index: str, progress: Callable[[str], None] | None = None) -> str:
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"RVC 입력 파일이 없습니다: {source}")
    with tempfile.TemporaryDirectory(prefix="rvc-chunks-", dir=target.parent) as temporary:
        workspace = Path(temporary)
        input_dir, output_dir = workspace / "input", workspace / "output"
        plans, source_rate, source_frames = prepare_chunks(source, input_dir)
        duration = source_frames / source_rate
        env = os.environ.copy()
        env.update({"PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
                    "CUDA_VISIBLE_DEVICES": gpu_index, "HF_HUB_OFFLINE": "1"})
        command = [str(python), str(ROOT / "scripts" / "rvc_batch.py"),
                   "--model", str(model), "--input-dir", str(input_dir),
                   "--output-dir", str(output_dir), "--pitch", str(pitch),
                   "--index-rate", str(index_rate), "--protect", str(protect)]
        if index_rate > 0 and index is not None:
            command.extend(["--index", str(index)])
        record("convert", "입력 분석", f"입력 {source}; 길이 {duration:.2f}초; "
               f"샘플레이트 {source_rate}; 구간 {len(plans)}개; 모델 {model}; 인덱스 {index}; "
               f"GPU {gpu_index or '자동/CPU'}; RMVPE; pitch {pitch}; protect {protect}; index rate {index_rate}")
        _run_batch(command, env=env, total=len(plans), progress=progress)
        merge_chunks(plans, output_dir, source_rate, source_frames, target)
    record("convert", "병합 완료", f"{len(plans)}개 구간; {target}; {sf.info(target).duration:.2f}초")
    return f"음색 변환: {target.name} · {len(plans)}개 구간"
