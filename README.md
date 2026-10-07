# AiChangingMusic

원곡의 가사 구간과 박자 단서를 분석해 개사 가사를 편집하고, MIDI/가이드 WAV 또는 로컬 AI 생성 WAV로 이어가는 Windows용 패러디 제작 보조 도구입니다. 처리는 기본적으로 이 PC에서 수행합니다.

> 이 앱은 한 곡의 목소리를 학습하거나 특정 인물의 음성을 복제하지 않습니다. 참고 음원은 ACE-Step의 스타일 조건 입력으로 사용되며, 출력 보컬이 원본과 같아진다는 보장은 없습니다. 본인 음성 또는 명시적 사용 허락을 받은 음원만 참고 입력으로 사용하세요.

## 빠른 시작

### 내려받기

ACE-Step 엔진을 함께 받으려면 서브모듈을 포함해 복제합니다.

```powershell
git clone --recurse-submodules https://github.com/illililiiililii/AiChangingMusic.git
cd AiChangingMusic
```

일반 `git clone`을 이미 했다면 다음을 한 번 실행합니다.

```powershell
git submodule update --init --recursive
```

### 필수 구성 요소

- Windows 10/11 및 Python 3.11 또는 3.12
- 로컬 음성 인식용 Shotcut 설치: 기본 경로에서 `ffmpeg.exe`, `whisper-cli.exe`, `ggml-base-q5_1.bin`을 찾습니다.
- AI 곡 생성용 ACE-Step Python 환경과 모델 파일. `ai_engine/README.md`의 설치 절차를 참고하세요.
- Node.js는 테스트를 실행할 때만 필요합니다.

Shotcut 실행 파일 또는 Whisper 모델이 기본 경로에 없다면 현재 `server.py`의 `FFMPEG_CANDIDATES`, `WHISPER_CANDIDATES`, `MODEL_CANDIDATES` 경로를 PC 환경에 맞게 수정해야 합니다.

### ACE-Step 설치

`ai_engine`은 ACE-Step 공식 저장소를 서브모듈로 고정해 사용합니다. Python 3.11/3.12 환경에서 `uv`를 설치한 다음:

```powershell
cd ai_engine
uv sync
cd ..
```

앱은 `ai_engine/.venv/Scripts/python.exe`와 `ai_engine/acestep/api_server.py`가 있을 때 생성 엔진을 사용할 수 있습니다. 모델은 첫 생성 때 내려받거나 초기화될 수 있으며, 저장 공간과 메모리가 필요합니다. 하드웨어·모델 설정의 최신 안내는 서브모듈 README를 기준으로 하세요.

### 실행

프로젝트 폴더에서 `시작.bat`을 실행하면 로컬 서버와 브라우저가 열립니다.

```text
http://127.0.0.1:8765/
```

직접 실행하려면:

```powershell
python server.py --open-browser
```

서버는 loopback 주소(`127.0.0.1`)에만 바인딩됩니다. 다른 포트를 쓰려면 PowerShell에서 실행 전에 설정합니다.

```powershell
$env:GASA_PORT = "8766"
python server.py --open-browser
```

서버 창을 닫으면 앱과 앱이 시작한 ACE-Step API 프로세스도 종료됩니다.

## 사용 순서

1. **원본 불러오기**: 단일 오디오/영상, 소스 폴더 또는 ZIP을 선택합니다. 폴더 선택은 Chrome/Edge의 디렉터리 선택 기능을 사용하며 지원 미디어를 최대 30개까지 순차 분석합니다.
2. **원본 후보 확인**: 폴더/ZIP은 인식된 한글 음절, 구간 수, 텍스트 중복도, 시간 범위를 바탕으로 후보 점수를 계산해 가장 높은 파일을 선택합니다. 점수는 휴리스틱이며 정확도 보증값은 아닙니다. 선택한 소스는 WAV로 정규화해 파형과 재생에 사용합니다.
3. **원곡 가사 교정**: `원곡 가사·음절 시점 분석`을 실행하고 Whisper/OCR 결과를 직접 교정합니다. 영상에 화면 가사가 있으면 OCR 결과를 우선 사용할 수 있습니다.
4. **개사 가사 배치**: 내 가사를 한 프레이즈씩 줄바꿈해 입력하고 `교정 가사로 내 음절 배치`를 누릅니다. 음절 수와 프레이즈 순서를 기준으로 원곡 구간에 매핑합니다.
5. **박자·피치 확인**: BPM 후보와 음높이 후보를 확인하고, 파형을 재생하면서 프레이즈 시작·음절 박·길이·MIDI 피치를 조정합니다. 자동 분석은 초안이므로 최종 결과를 직접 들어 확인하세요.
6. **내보내기**: MIDI, 원곡에 삼각파 가이드를 섞은 WAV, 또는 프로젝트 JSON을 저장합니다. 프로젝트 JSON에는 오디오 자체가 들어 있지 않으므로 다시 편집할 때 같은 음원을 선택해야 합니다.
7. **AI 생성**: 가사, 편곡 설명, BPM, 길이를 설정합니다. 원곡을 스타일 참고로 쓸 때 동의 체크가 필요합니다. 결과는 ACE-Step이 새로 생성하며, 원곡 목소리를 재현하지 않습니다.

## 처리 방식과 한계

- 브라우저는 편집 UI이고 `server.py`가 loopback HTTP 서버/API입니다. 오디오 업로드, FFmpeg 변환, Whisper/OCR, 파일 저장, ACE-Step API 연결은 로컬에서 수행합니다.
- Whisper는 Shotcut의 `whisper-cli`와 `ggml-base-q5_1.bin`을 사용합니다. 영상은 Windows 한국어 OCR을 시도하고, 실패하거나 가사가 없으면 음성 인식으로 처리합니다.
- OCR은 초당 한 프레임을 읽으므로 화면 전환 시점은 실제 보컬 음절 시점과 다를 수 있습니다. Whisper도 노래 가사를 틀리게 인식할 수 있으니 교정이 필요합니다.
- 박자 후보는 오디오 에너지 변화의 반복을 이용한 근사값입니다. 음높이는 혼합 음원에서 추정하므로 반주를 보컬로 오인할 수 있습니다.
- MIDI와 가이드 WAV는 편집 보조물입니다. 가이드 WAV의 신스는 사람 목소리가 아닙니다.
- ACE-Step 참조 입력은 모델의 스타일 생성 조건이지 한 곡으로 개인 음성 모델을 학습하는 기능이 아닙니다. 음성 클로닝 또는 특정 인물과 동일한 목소리 생성 기능은 제공하지 않습니다.
- 업로드 최대 크기는 2GiB입니다. ZIP은 압축 해제 후 5GiB 및 10,000개 파일까지 허용하며, 경로 탈출과 심볼릭 링크를 거부합니다. ZIP 분석 중 선택된 WAV는 임시 보관되며 서버 종료 때 삭제됩니다.
- 내보낸 파일은 `exports/`에 남습니다. 이 폴더와 모델 체크포인트/가상환경은 Git에 포함되지 않습니다.

## 프로젝트 구조

```text
.
├── 시작.bat                 # Windows 실행기
├── server.py                # 로컬 HTTP API, 분석/변환, 저장, AI 엔진 연결
├── index.html               # 한국어 편집기 화면
├── app.js                   # 파형, 타이밍, MIDI/WAV 내보내기, 소스 선택
├── style.css                # 편집기 스타일
├── lyrics_structure.js      # 반복 가사 구절 기반 구조 태거
├── ocr_video.ps1            # Windows Media OCR 연결
├── tests/                   # Node 및 Python 회귀 테스트
├── exports/                 # 로컬 생성 파일 (Git 제외)
└── ai_engine/               # ACE-Step 공식 코드 서브모듈
```

## 테스트

Python 서버/ZIP 처리 테스트:

```powershell
python -m unittest tests.test_source_bundle -v
```

프론트엔드 및 가사 구조 테스트:

```powershell
node --test tests/*.test.cjs
```

일반 사용에는 Node.js가 필요하지 않습니다.
