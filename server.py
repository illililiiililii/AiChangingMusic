# ---------------------  server.py (전체)  ---------------------
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import atexit, json, mimetypes, os, re, shutil, subprocess, sys, tempfile, threading, time, urllib.parse, urllib.request, uuid, webbrowser, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FFMPEG_CANDIDATES = [shutil.which('ffmpeg'), r'C:\Program Files\Shotcut\ffmpeg.exe', r'C:\Program Files (x86)\kdisk.co.kr\ffmpeg.exe']
WHISPER_CANDIDATES = [shutil.which('whisper-cli'), r'C:\Program Files\Shotcut\whisper-cli.exe']
MODEL_CANDIDATES = [r'C:\Program Files\Shotcut\share\shotcut\whisper_models\ggml-base-q5_1.bin']
POWERSHELL = r'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe'
EXPORT_DIR = ROOT / 'exports'
AI_DIR = ROOT / 'ai_engine'
AI_API_SCRIPT = AI_DIR / 'acestep' / 'api_server.py'
AI_PYTHON = AI_DIR / '.venv' / 'Scripts' / 'python.exe'
AI_API_URL = 'http://127.0.0.1:8001'
AI_TEMP_DIR = Path(tempfile.mkdtemp(prefix='gasa-beat-ai-'))
SOURCE_TEMP_DIR = Path(tempfile.mkdtemp(prefix='gasa-beat-source-'))
AI_API_PROCESS = None
AI_JOBS = {}
AI_JOBS_LOCK = threading.Lock()
SOURCE_FILES = {}
SOURCE_FILES_LOCK = threading.Lock()
MAX_UPLOAD_BYTES = 2 * 1024 * 1024 * 1024
MAX_ARCHIVE_FILES = 10000
MAX_ARCHIVE_UNPACKED_BYTES = 5 * 1024 * 1024 * 1024

def first_existing(items):
    for item in items:
        if item and Path(item).is_file():
            return str(Path(item))
    return None

class ApiError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def _run_checked(command, timeout=1800):
    result = subprocess.run(command, check=False, capture_output=True, timeout=timeout)
    if result.returncode:
        detail = result.stderr.decode('utf-8', errors='replace').strip()
        raise RuntimeError(detail[-2000:] or f'명령 실행 실패: {Path(command[0]).name}')
    return result


def _audio_units(text, start, end):
    syllables = [char for char in text if '\uac00' <= char <= '\ud7a3']
    if not syllables:
        syllables = [char for char in text if not char.isspace()]
    if not syllables:
        return []
    duration = max(0.0, end - start)
    return [
        {'start': start + duration * index / len(syllables),
         'end': start + duration * (index + 1) / len(syllables)}
        for index in range(len(syllables))
    ]


def _normalize_transcription(items):
    segments = []
    for item in items:
        offsets = item.get('offsets') or {}
        start = float(offsets.get('from', item.get('start', 0) * 1000)) / 1000
        end = float(offsets.get('to', item.get('end', start) * 1000)) / 1000
        text = str(item.get('text', '')).strip()
        if not text:
            continue
        segments.append({
            'start': start,
            'end': max(start, end),
            'text': text,
            'tokens': item.get('tokens', []),
            'units': _audio_units(text, start, max(start, end)),
        })
    return segments


def _video_lyrics(video_path, work_dir, ffmpeg):
    frames_dir = work_dir / 'frames'
    frames_dir.mkdir()
    _run_checked([
        ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-i', str(video_path),
        '-vf', 'fps=1', '-q:v', '4', str(frames_dir / 'frame_%04d.jpg'),
    ])
    ocr_script = ROOT / 'ocr_video.ps1'
    if not Path(POWERSHELL).is_file() or not ocr_script.is_file():
        return []
    result = _run_checked([
        POWERSHELL, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(ocr_script),
        '-FramesDir', str(frames_dir),
    ], timeout=600)
    rows = json.loads(result.stdout.decode('utf-8-sig'))
    if isinstance(rows, dict):
        rows = [rows]
    segments = []
    previous_text = None
    for row in rows:
        text = str(row.get('text', '')).strip()
        if not text or text == previous_text:
            continue
        start = max(0.0, float(row.get('index', 1)) - 1)
        segments.append({'start': start, 'end': start + 1.0, 'text': text, 'tokens': []})
        previous_text = text
    for index, segment in enumerate(segments[:-1]):
        segment['end'] = max(segment['start'] + 0.1, segments[index + 1]['start'])
    for segment in segments:
        segment['units'] = _audio_units(segment['text'], segment['start'], segment['end'])
    return segments


def _zip_source_candidates(source_dir):
    candidates = []
    audio_ext = {'.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac', '.aiff', '.aif', '.webm'}
    video_ext = {'.mp4', '.mkv', '.mov', '.avi', '.webm'}
    for path in sorted(source_dir.rglob('*'), key=lambda item: str(item).lower()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in audio_ext or suffix in video_ext:
            priority = 0 if suffix in video_ext else 1
            candidates.append((priority, path))
    candidates.sort(key=lambda item: (item[0], str(item[1]).lower()))
    return [path for _, path in candidates]


def _transcription_quality(result):
    segments = [
        segment for segment in result.get('segments', [])
        if isinstance(segment, dict) and str(segment.get('text', '')).strip()
    ]
    if not segments:
        return 0.0
    korean_chars = sum(
        1 for segment in segments for char in str(segment['text'])
        if '\uac00' <= char <= '\ud7a3'
    )
    unique_texts = {
        re.sub(r'\s+', '', str(segment['text'])).casefold()
        for segment in segments
    }
    starts = [float(segment.get('start', 0) or 0) for segment in segments]
    ends = [float(segment.get('end', start) or start) for segment, start in zip(segments, starts)]
    coverage = max(0.0, max(ends) - min(starts))
    uniqueness = len(unique_texts) / len(segments)
    return korean_chars + min(len(segments), 40) * 3 + uniqueness * 12 + min(coverage, 180) / 30


def _retain_source_file(source_path):
    source_id = uuid.uuid4().hex
    destination = SOURCE_TEMP_DIR / f'{source_id}.wav'
    ffmpeg = first_existing(FFMPEG_CANDIDATES)
    if not ffmpeg:
        raise ApiError('선택된 ZIP 음원을 준비할 FFmpeg를 찾지 못했습니다.', 503)
    try:
        _run_checked([
            ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-i', str(source_path),
            '-vn', '-ac', '1', '-ar', '44100', '-c:a', 'pcm_s16le', str(destination),
        ])
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    with SOURCE_FILES_LOCK:
        SOURCE_FILES[source_id] = destination
        while len(SOURCE_FILES) > 5:
            oldest_id = next(iter(SOURCE_FILES))
            SOURCE_FILES.pop(oldest_id).unlink(missing_ok=True)
    return source_id


def transcribe_media(source_path, extension):
    ffmpeg = first_existing(FFMPEG_CANDIDATES)
    whisper = first_existing(WHISPER_CANDIDATES)
    model = first_existing(MODEL_CANDIDATES)
    missing = [name for name, value in [('FFmpeg', ffmpeg), ('Whisper', whisper), ('Whisper 모델', model)] if not value]
    if missing:
        raise ApiError(f'{", ".join(missing)}을(를) 찾지 못했습니다. Shotcut 설치 경로를 확인해 주세요.', 503)
    with tempfile.TemporaryDirectory(prefix='gasa-transcribe-') as temp_name:
        work_dir = Path(temp_name)
        source = work_dir / f'input{extension}'
        shutil.copyfile(source_path, source)
        if extension in {'.mp4', '.mkv', '.mov', '.webm', '.avi'}:
            try:
                segments = _video_lyrics(source, work_dir, ffmpeg)
                if segments:
                    return {'segments': segments, 'sourceMethod': 'video-lyrics-ocr', 'model': 'Windows 한국어 OCR'}
            except (OSError, RuntimeError, json.JSONDecodeError, subprocess.SubprocessError):
                pass
        wav_path = work_dir / 'audio.wav'
        _run_checked([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-i', str(source), '-ac', '1', '-ar', '16000', str(wav_path)])
        output_base = work_dir / 'transcription'
        threads = str(max(2, min(8, os.cpu_count() or 4)))
        _run_checked([
            whisper, '-m', model, '-l', 'ko', '-ng', '-t', threads,
            '-dtw', 'base', '-ojf', '-of', str(output_base), str(wav_path),
        ])
        output_path = output_base.with_suffix('.json')
        if not output_path.is_file():
            raise RuntimeError('Whisper 결과 JSON 파일을 찾지 못했습니다.')
        with output_path.open('r', encoding='utf-8') as source_file:
            payload = json.load(source_file)
        items = payload.get('transcription', payload.get('segments', []))
        return {
            'segments': _normalize_transcription(items),
            'sourceMethod': 'whisper-dtw',
            'model': Path(model).name,
        }


def transcribe_bundle(bundle_path):
    with tempfile.TemporaryDirectory(prefix='gasa-source-bundle-') as temp_name:
        bundle_dir = Path(temp_name)
        with zipfile.ZipFile(bundle_path) as archive:
            entries = archive.infolist()
            if len(entries) > MAX_ARCHIVE_FILES:
                raise ApiError('ZIP 안의 파일이 너무 많습니다. 10,000개 이하로 정리해 주세요.', 413)
            unpacked_size = sum(entry.file_size for entry in entries)
            if unpacked_size > MAX_ARCHIVE_UNPACKED_BYTES:
                raise ApiError('압축 해제 용량이 5GB를 초과합니다.', 413)
            for entry in entries:
                target = (bundle_dir / entry.filename).resolve()
                if not target.is_relative_to(bundle_dir.resolve()):
                    raise ApiError('ZIP에 허용되지 않는 경로가 포함되어 있습니다.', 400)
                if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ApiError('ZIP 심볼릭 링크는 분석할 수 없습니다.', 400)
            archive.extractall(bundle_dir)
        candidate_files = _zip_source_candidates(bundle_dir)
        if not candidate_files:
            raise ApiError('소스 ZIP/폴더에서 분석 가능한 오디오 또는 영상 파일을 찾지 못했습니다.', 400)
        best_result = None
        best_source = None
        best_score = -1.0
        for candidate in candidate_files:
            try:
                result = transcribe_media(candidate, candidate.suffix.lower())
                if not result.get('segments'):
                    continue
                score = _transcription_quality(result)
                if score > best_score:
                    best_result = result
                    best_source = candidate
                    best_score = score
            except (ApiError, OSError, RuntimeError, json.JSONDecodeError, subprocess.SubprocessError, ValueError):
                continue
        if best_result is None:
            raise ApiError('소스 ZIP 안의 음원 파일을 분석할 수 없습니다. 오디오/영상 파일 형식을 확인해 주세요.', 400)
        source_id = _retain_source_file(best_source)
        return {
            **best_result,
            'sourceMethod': best_result.get('sourceMethod', 'source-bundle'),
            'model': best_result.get('model', 'ZIP 소스 분석'),
            'sourceFile': {
                'id': source_id,
                'name': best_source.name,
                'url': f'/api/source/{source_id}',
                'qualityScore': round(best_score, 1),
            },
        }


def _send_api_json(handler, payload, status=200):
    body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    handler.send_response(status)
    handler.send_header('Content-Type', 'application/json; charset=utf-8')
    handler.send_header('Content-Length', str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json(handler):
    length = int(handler.headers.get('Content-Length', '0'))
    if length <= 0 or length > 2 * 1024 * 1024:
        raise ApiError('요청 JSON 크기가 올바르지 않습니다.')
    try:
        return json.loads(handler.rfile.read(length).decode('utf-8'))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ApiError('JSON 요청을 읽을 수 없습니다.') from error


def _request_extension(handler):
    value = urllib.parse.parse_qs(urllib.parse.urlsplit(handler.path).query).get('ext', ['.wav'])[0].lower()
    if not re.fullmatch(r'\.[a-z0-9]{1,8}', value):
        raise ApiError('파일 확장자가 올바르지 않습니다.')
    return value


def _store_request_body(handler, destination):
    length = int(handler.headers.get('Content-Length', '0'))
    if length <= 0 or length > MAX_UPLOAD_BYTES:
        raise ApiError('업로드 파일 크기가 올바르지 않습니다.', 413)
    remaining = length
    with Path(destination).open('wb') as output:
        while remaining:
            chunk = handler.rfile.read(min(1024 * 1024, remaining))
            if not chunk:
                raise ApiError('업로드가 중간에 끊겼습니다.', 400)
            output.write(chunk)
            remaining -= len(chunk)


def _safe_filename(value, fallback):
    name = Path(value or '').name
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', name).strip(' .')
    return (name or fallback)[:180]


def _api_request(path, payload=None, timeout=30):
    data = None if payload is None else json.dumps(payload).encode('utf-8')
    headers = {'Content-Type': 'application/json'} if data is not None else {}
    api_request = urllib.request.Request(AI_API_URL + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(api_request, timeout=timeout) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as error:
        detail = error.read().decode('utf-8', errors='replace')
        raise RuntimeError(detail[:1000] or f'ACE-Step API 오류 ({error.code})') from error


def _ensure_ai_api():
    global AI_API_PROCESS
    try:
        _api_request('/health', timeout=2)
        return
    except (OSError, RuntimeError, ValueError):
        pass
    if not AI_PYTHON.is_file() or not AI_API_SCRIPT.is_file():
        raise RuntimeError('ai_engine/.venv 또는 ACE-Step API 서버를 찾지 못했습니다.')
    if AI_API_PROCESS is not None and AI_API_PROCESS.poll() is not None:
        AI_API_PROCESS = None
    if AI_API_PROCESS is None:
        environment = os.environ.copy()
        environment.setdefault('ACESTEP_DEVICE', 'cpu')
        environment.setdefault('ACESTEP_LM_DEVICE', 'cpu')
        environment.setdefault('ACESTEP_INIT_LLM', 'false')
        AI_API_PROCESS = subprocess.Popen(
            [str(AI_PYTHON), str(AI_API_SCRIPT), '--host', '127.0.0.1', '--port', '8001'],
            cwd=AI_DIR, env=environment,
            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
        )
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        if AI_API_PROCESS.poll() is not None:
            AI_API_PROCESS = None
            raise RuntimeError('ACE-Step API 프로세스가 시작 중 종료됐습니다. ai_engine 로그를 확인해 주세요.')
        try:
            _api_request('/health', timeout=2)
            return
        except (OSError, RuntimeError, ValueError):
            time.sleep(0.5)
    raise RuntimeError('ACE-Step API가 180초 안에 시작되지 않았습니다.')


def stop_ai_api():
    global AI_API_PROCESS
    process = AI_API_PROCESS
    AI_API_PROCESS = None
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def cleanup_ai_reference():
    shutil.rmtree(AI_TEMP_DIR, ignore_errors=True)


def cleanup_source_files():
    shutil.rmtree(SOURCE_TEMP_DIR, ignore_errors=True)


def _ai_status():
    installed = AI_PYTHON.is_file() and AI_API_SCRIPT.is_file()
    try:
        _api_request('/health', timeout=1)
        api_running = True
    except (OSError, RuntimeError, ValueError):
        api_running = False
    with AI_JOBS_LOCK:
        generating = any(job['status'] == 'running' for job in AI_JOBS.values())
    return {'installed': installed, 'api': api_running, 'generating': generating}


def _download_generated_audio(file_value, job_id):
    if not file_value:
        raise RuntimeError('ACE-Step 작업은 완료됐지만 오디오 파일 경로가 없습니다.')
    if file_value.startswith('/'):
        audio_url = AI_API_URL + file_value
        with urllib.request.urlopen(audio_url, timeout=120) as response:
            name = f'ai_song_{job_id[:8]}.wav'
            target = EXPORT_DIR / name
            with target.open('wb') as output:
                shutil.copyfileobj(response, output)
    elif file_value.startswith('http://') or file_value.startswith('https://'):
        with urllib.request.urlopen(file_value, timeout=120) as response:
            name = f'ai_song_{job_id[:8]}.wav'
            target = EXPORT_DIR / name
            with target.open('wb') as output:
                shutil.copyfileobj(response, output)
    elif Path(file_value).is_file():
        name = f'ai_song_{job_id[:8]}.wav'
        target = EXPORT_DIR / name
        shutil.copyfile(file_value, target)
    else:
        raise RuntimeError('ACE-Step에서 생성된 오디오 파일을 가져오지 못했습니다.')
    return name, '/exports/' + urllib.parse.quote(name)


def _run_ai_job(job_id, request_data):
    reference_path = request_data.get('referenceAudioPath') or None
    try:
        _ensure_ai_api()
        ace_request = {
            'prompt': str(request_data.get('prompt', '')).strip(),
            'lyrics': str(request_data.get('lyrics', '')).strip(),
            'thinking': False,
            'bpm': max(40, min(240, int(request_data.get('bpm') or 120))),
            'time_signature': f"{max(1, min(12, int(request_data.get('meter') or 4)))}/4",
            'vocal_language': 'ko',
            'audio_duration': max(10, min(600, float(request_data.get('duration') or 40))),
            'reference_audio_path': reference_path,
            'audio_format': 'wav',
            'inference_steps': 8,
        }
        response = _api_request('/release_task', ace_request, timeout=60)
        response_data = response.get('data', response)
        task_id = response_data.get('task_id')
        if not task_id:
            raise RuntimeError(response.get('message') or 'ACE-Step이 생성 작업 ID를 반환하지 않았습니다.')
        with AI_JOBS_LOCK:
            AI_JOBS[job_id]['progressText'] = 'AI 생성 대기열에 등록됨'
        deadline = time.monotonic() + 6 * 60 * 60
        while time.monotonic() < deadline:
            result_payload = _api_request('/query_result', {'task_id_list': [task_id]}, timeout=30)
            records = result_payload.get('data') or []
            record = records[0] if records else {}
            task_status = record.get('status', 0)
            if task_status == 1:
                result_data = record.get('result', [])
                if isinstance(result_data, str):
                    result_data = json.loads(result_data)
                audio = result_data[0] if result_data else {}
                name, url = _download_generated_audio(audio.get('file'), job_id)
                with AI_JOBS_LOCK:
                    AI_JOBS[job_id].update({'status': 'done', 'name': name, 'url': url, 'progress': 100, 'progressText': '생성 완료'})
                return
            if task_status == 2:
                detail = record.get('result', '')
                raise RuntimeError(str(detail)[:1000] or 'ACE-Step 생성 작업이 실패했습니다.')
            try:
                progress_data = json.loads(record.get('result', '[]'))
                current = progress_data[0] if progress_data else {}
            except (TypeError, json.JSONDecodeError, IndexError):
                current = {}
            progress = float(current.get('progress', 0) or 0)
            if 0 < progress <= 1:
                progress *= 100
            with AI_JOBS_LOCK:
                AI_JOBS[job_id].update({
                    'progress': min(99, progress),
                    'progressText': str(record.get('progress_text') or current.get('stage') or '모델이 곡을 생성하고 있습니다')[:200],
                })
            time.sleep(2)
        raise RuntimeError('AI 곡 생성 시간이 제한 시간을 초과했습니다.')
    except Exception as error:
        with AI_JOBS_LOCK:
            if job_id in AI_JOBS:
                AI_JOBS[job_id].update({'status': 'error', 'error': str(error), 'progressText': ''})
    finally:
        if reference_path:
            path = Path(reference_path).resolve()
            if path.parent == AI_TEMP_DIR.resolve():
                path.unlink(missing_ok=True)


class Handler(SimpleHTTPRequestHandler):
    def _json(self, payload, status=200):
        _send_api_json(self, payload, status)

    def _error(self, error):
        if isinstance(error, ApiError):
            self._json({'error': str(error)}, error.status)
        else:
            self._json({'error': str(error) or '요청 처리에 실패했습니다.'}, 500)

    def do_GET(self):
        route = urllib.parse.urlsplit(self.path)
        if route.path == '/api/status':
            self._json({
                'ready': all(first_existing(items) for items in (FFMPEG_CANDIDATES, WHISPER_CANDIDATES, MODEL_CANDIDATES)),
                'ffmpeg': bool(first_existing(FFMPEG_CANDIDATES)),
                'whisper': bool(first_existing(WHISPER_CANDIDATES)),
                'model': bool(first_existing(MODEL_CANDIDATES)),
            })
            return
        if route.path == '/api/ai-status':
            self._json(_ai_status())
            return
        if route.path == '/api/ai-job':
            job_id = urllib.parse.parse_qs(route.query).get('id', [''])[0]
            with AI_JOBS_LOCK:
                job = AI_JOBS.get(job_id)
                payload = dict(job) if job else None
            if payload is None:
                self._json({'error': 'AI 작업을 찾을 수 없습니다.'}, 404)
            else:
                self._json(payload)
            return
        if route.path.startswith('/api/source/'):
            source_id = route.path.rsplit('/', 1)[-1]
            if not re.fullmatch(r'[a-f0-9]{32}', source_id):
                self._json({'error': '소스 파일 ID가 올바르지 않습니다.'}, 400)
                return
            with SOURCE_FILES_LOCK:
                source_path = SOURCE_FILES.get(source_id)
            if not source_path or not source_path.is_file():
                self._json({'error': '선택된 소스 파일이 만료되었습니다. ZIP을 다시 분석해 주세요.'}, 404)
                return
            self.send_response(200)
            self.send_header('Content-Type', 'audio/wav')
            self.send_header('Content-Length', str(source_path.stat().st_size))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            with source_path.open('rb') as source_file:
                shutil.copyfileobj(source_file, self.wfile)
            return
        super().do_GET()

    def do_POST(self):
        route = urllib.parse.urlsplit(self.path)
        try:
            if route.path in {'/api/save-midi', '/api/save-project', '/api/save-audio'}:
                self._save_export(route.path, route.query)
            elif route.path == '/api/transcribe':
                self._transcribe(route.query)
            elif route.path == '/api/source':
                self._retain_uploaded_source(route.query)
            elif route.path == '/api/ai-reference':
                self._prepare_reference(route.query)
            elif route.path == '/api/ai-generate':
                self._start_generation()
            elif route.path == '/api/ai-stop':
                stop_ai_api()
                self._json({'message': '로컬 AI 엔진을 종료했습니다.'})
            else:
                self.send_error(404)
        except (ApiError, OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            self._error(error)

    def _save_export(self, route, query):
        requested = urllib.parse.parse_qs(query).get('name', [''])[0]
        fallback = {'/api/save-midi': '가사박자.mid', '/api/save-project': '가사박자.gasa.json', '/api/save-audio': '가이드.wav'}[route]
        name = _safe_filename(requested, fallback)
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        temporary = EXPORT_DIR / f'.{uuid.uuid4().hex}.upload'
        try:
            _store_request_body(self, temporary)
            os.replace(temporary, EXPORT_DIR / name)
        finally:
            temporary.unlink(missing_ok=True)
        self._json({'name': name, 'url': '/exports/' + urllib.parse.quote(name)})

    def _transcribe(self, query):
        extension = _request_extension(self)
        with tempfile.NamedTemporaryFile(prefix='gasa-upload-', suffix=extension, delete=False) as upload:
            upload_path = Path(upload.name)
        try:
            _store_request_body(self, upload_path)
            if extension == '.zip':
                self._json(transcribe_bundle(upload_path))
            else:
                result = transcribe_media(upload_path, extension)
                self._json({**result, 'qualityScore': round(_transcription_quality(result), 1)})
        finally:
            upload_path.unlink(missing_ok=True)

    def _retain_uploaded_source(self, query):
        extension = _request_extension(self)
        with tempfile.NamedTemporaryFile(prefix='gasa-source-', suffix=extension, delete=False) as upload:
            upload_path = Path(upload.name)
        try:
            _store_request_body(self, upload_path)
            source_id = _retain_source_file(upload_path)
        finally:
            upload_path.unlink(missing_ok=True)
        self._json({'id': source_id, 'url': f'/api/source/{source_id}'})

    def _prepare_reference(self, query):
        extension = _request_extension(self)
        ffmpeg = first_existing(FFMPEG_CANDIDATES)
        if not ffmpeg:
            raise ApiError('참고 음원을 변환할 FFmpeg를 찾지 못했습니다.', 503)
        with tempfile.NamedTemporaryFile(prefix='gasa-reference-', suffix=extension, delete=False) as upload:
            upload_path = Path(upload.name)
        output_path = AI_TEMP_DIR / f'reference-{uuid.uuid4().hex}.wav'
        try:
            _store_request_body(self, upload_path)
            _run_checked([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-i', str(upload_path), '-vn', '-ac', '1', '-ar', '44100', str(output_path)])
        finally:
            upload_path.unlink(missing_ok=True)
        self._json({'path': str(output_path)})

    def _start_generation(self):
        request_data = _read_json(self)
        reference_path = request_data.get('referenceAudioPath')
        if reference_path and request_data.get('voiceConsent') is not True:
            raise ApiError('참고 음원 사용 동의가 필요합니다.', 403)
        if not str(request_data.get('lyrics', '')).strip():
            raise ApiError('가사를 입력해 주세요.')
        if reference_path:
            resolved = Path(reference_path).resolve()
            if resolved.parent != AI_TEMP_DIR.resolve() or not resolved.is_file():
                raise ApiError('참고 음원 경로가 유효하지 않습니다.')
        with AI_JOBS_LOCK:
            if any(job['status'] == 'running' for job in AI_JOBS.values()):
                raise ApiError('이미 곡 생성 작업이 진행 중입니다.', 409)
            job_id = uuid.uuid4().hex
            AI_JOBS[job_id] = {'id': job_id, 'status': 'running', 'progress': 0, 'progressText': 'AI 엔진을 준비하고 있습니다'}
        worker = threading.Thread(target=_run_ai_job, args=(job_id, request_data), daemon=True)
        worker.start()
        self._json({'id': job_id, 'status': 'running'}, 202)


def main():
    os.chdir(ROOT)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    port = int(os.environ.get('GASA_PORT', '8765'))
    url = f'http://127.0.0.1:{port}/'
    print(f'가사박자 is running at {url}', flush=True)
    print('이 창을 닫으면 도구가 종료됩니다.', flush=True)
    if '--open-browser' in sys.argv:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    ThreadingHTTPServer(('127.0.0.1', port), Handler).serve_forever()


atexit.register(cleanup_ai_reference)
atexit.register(cleanup_source_files)
atexit.register(stop_ai_api)

if __name__ == '__main__':
    main()