# Windows 11 설치 안내

## 1. 선행 설치

- [Python 3.10 x64](https://www.python.org/downloads/windows/) 및 [Python 3.12 x64](https://www.python.org/downloads/windows/): 두 버전을 설치합니다. `py -3.10`과 `py -3.12`가 작동하거나 기본 설치 위치에 있어야 합니다.
- [Git for Windows](https://git-scm.com/download/win): 설치 스크립트가 RVC 저장소를 가져오는 데 사용합니다.
- NVIDIA GPU를 사용할 때 [NVIDIA 드라이버](https://www.nvidia.com/Download/index.aspx): 드라이버 설치는 이 프로젝트가 자동으로 수행하지 않습니다.
- 약 12GB 이상의 여유 공간과 최초 다운로드를 위한 인터넷.

관리자 권한이나 시스템 Python 변경 없이 프로젝트 내부 `.venv`(Python 3.10), `.venv-rvc`(Python 3.12)를 만듭니다. 설치 스크립트는 Torch CUDA 11.8 휠, Python 패키지, RVC 코드, FFmpeg, DiffSingerKR, RVC 자산, Demucs, faster-whisper 모델을 내려받습니다. 모델 가중치는 Git에 올라가지 않습니다.

## 2. 자동 설치

저장소 폴더에서 `setup.bat`을 실행합니다. 오류가 나면 창에 오류가 남습니다. 설치 완료 후 `run.bat`을 실행합니다. PowerShell에서는 다음과 같습니다.

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\setup.ps1
.\run.bat
```

`setup.ps1`은 이미 받은 파일을 재사용합니다. 설치 뒤 `.venv\Scripts\python.exe scripts\check_environment.py`로 두 GPU와 모델 위치를 확인할 수 있습니다. 실제 로컬 추론은 모델이 캐시된 뒤 오프라인에서 작동합니다.

## 3. 로컬 자동 개사 설정 (선택)

수동 가사는 Ollama 없이 사용할 수 있습니다. 자동 개사에는 [Ollama for Windows](https://ollama.com/download/windows)를 따로 설치하고 실행한 다음, 터미널에서 `ollama pull qwen3.5:4b`를 한 번 실행합니다. 앱은 `127.0.0.1:11434/api/generate`에만 접속합니다. 다른 로컬 Ollama 모델을 쓰려면 가사 탭에서 이름을 바꾸세요. 모델 다운로드 후 오프라인 로컬 서버로 사용할 수 있습니다.

## 4. 음성 프로필

목소리 탭의 허락 확인을 체크하고, 동일 화자의 깨끗한 일반 말하기 오디오 10분 이상이 있는 폴더를 선택해 학습합니다. MP3/WAV/FLAC/OGG/M4A를 읽습니다. 결과는 `external/RVC/assets/weights`, `external/RVC/assets/indices`에 생기고, 학습 기록은 `voice_data`에 남습니다. 6GB GPU에서는 batch 2, FP32로 순차 처리합니다. 다른 사용자의 허락받은 녹음도 동일한 절차로 가능합니다.

## 5. 문제 해결

- **Python 버전 없음**: 두 버전을 x64로 설치하고 `py -0p`를 확인합니다. 설치 스크립트는 기본 사용자 설치 위치도 검사합니다.
- **Git 없음**: Git for Windows 설치 후 새 터미널에서 재실행합니다.
- **CUDA 없음**: `scripts/check_environment.py`의 GPU 결과를 확인하고 NVIDIA 드라이버를 점검합니다. CPU 추론은 모델별로 느리며 전체 경로를 검증하지 않았습니다.
- **GPU 메모리 부족**: 다른 GPU 작업을 닫고 다시 실행합니다. Demucs는 CUDA 메모리 오류 시 CPU로 한 번 재시도합니다. RVC 학습은 6GB GPU 배치 2 설정입니다.
- **가사나 음정 오류**: 노래 ASR과 음정 추정은 초안입니다. 원곡 탭의 가사·음표 표를 수정한 뒤 정렬부터 재실행합니다.
- **음성 학습 실패**: `voice_data/<프로필>/training.log`의 마지막 오류를 확인합니다. 10분 이상의 깨끗한 단일 화자 녹음이 필요합니다.
- **FFmpeg·모델 없음**: `setup.bat`을 다시 실행합니다. 라이선스·다운로드 주소는 [MODELS.md](MODELS.md), [DEPENDENCIES.md](DEPENDENCIES.md)를 보세요.

신규 Windows PC에서 처음부터 설치하는 절차는 아직 별도 기기에서 검증하지 못했습니다. 이 개발 PC에서 설치와 실행, 모델 다운로드 및 로컬 전체 파이프라인을 시험했습니다.
