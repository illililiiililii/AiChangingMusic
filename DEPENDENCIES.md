# 데스크톱 앱 의존성

| 도구 | 목적 | 공식 설치·저장소 | 버전/라이선스 | 필수 여부·경로·API |
| --- | --- | --- | --- | --- |
| Python | 로컬 실행 | [python.org](https://www.python.org/downloads/release/python-31011/) / [cpython](https://github.com/python/cpython) | 3.10 x64, PSF | 필수. `py -3.10`, `.venv/Scripts/python.exe` |
| PyTorch | CUDA SVS 추론 | [공식 wheel 인덱스](https://download.pytorch.org/whl/cu118) / [pytorch](https://github.com/pytorch/pytorch) | 2.2.2+cu118, BSD 계열 | 필수. `.venv` 내부. 외부 API 없음 |
| DiffSingerKR | 모델 코드 | [CODEJIN GitHub](https://github.com/CODEJIN/DiffSingerKR) / [공식 Space](https://huggingface.co/spaces/codejin/diffsingerkr) | Space 고정 리비전, 코드 MIT | 필수. `external/DiffSingerKR-space/`; 외부 API 없음 |
| NumPy | 배열 | [PyPI](https://pypi.org/project/numpy/) / [GitHub](https://github.com/numpy/numpy) | 1.23.5, BSD | 필수. `.venv`, 외부 API 없음 |
| SciPy | 신호 처리 | [PyPI](https://pypi.org/project/scipy/) / [GitHub](https://github.com/scipy/scipy) | 1.10.1, BSD-3-Clause | 필수. `.venv`, 외부 API 없음 |
| librosa | 음향 함수·피치 평가 | [PyPI](https://pypi.org/project/librosa/) / [GitHub](https://github.com/librosa/librosa) | 0.9.2, ISC | 필수. `.venv`, 외부 API 없음 |
| matplotlib | 원본 추론 코드 import | [PyPI](https://pypi.org/project/matplotlib/) / [GitHub](https://github.com/matplotlib/matplotlib) | 3.7.5, PSF | 필수. `.venv`, 외부 API 없음 |
| PyYAML | 모델 설정 | [PyPI](https://pypi.org/project/PyYAML/) / [GitHub](https://github.com/yaml/pyyaml) | 6.0.2, MIT | 필수. `.venv`, 외부 API 없음 |
| tqdm | 추론 코드 import | [PyPI](https://pypi.org/project/tqdm/) / [GitHub](https://github.com/tqdm/tqdm) | 4.67.1, MPL-2.0 및 MIT | 필수. `.venv`, 외부 API 없음 |
| hgtk | 한글 자소 분해 | [PyPI](https://pypi.org/project/hgtk/) / [GitHub](https://github.com/bluedisk/hangul-toolkit) | 0.2.1, Apache-2.0 | 필수. `.venv`, 외부 API 없음 |
| mido | 원본 추론 코드 import | [PyPI](https://pypi.org/project/mido/) / [GitHub](https://github.com/mido/mido) | 1.3.3, MIT | 필수. `.venv`, 외부 API 없음 |
| soundfile | WAV 저장·검증 | [PyPI](https://pypi.org/project/soundfile/) / [GitHub](https://github.com/bastibe/python-soundfile) | 0.12.1, BSD-3-Clause | 필수. `.venv`, 외부 API 없음 |
| NVIDIA 드라이버 | GPU 실행 | [NVIDIA 다운로드](https://www.nvidia.com/Download/index.aspx) | 설치 GPU에 호환되는 버전 | GPU 사용 시 필수. 시스템 드라이버는 스크립트가 변경하지 않음 |

## 2단계 음색 변환 추가 도구

| 도구 | 용도 | 출처 | 이 검증 환경 |
| --- | --- | --- | --- |
| Python 3.12 x64 | RVC 독립 가상환경 | [python.org](https://www.python.org/downloads/) | `.venv-rvc` |
| PyTorch·Torchaudio CUDA 11.8 | RVC CUDA 학습·추론 | [공식 wheel 인덱스](https://download.pytorch.org/whl/cu118) | 두 패키지 모두 2.7.1+cu118 |
| RVC | 음성 전처리, 화자 학습, 노래 음색 변환 | [공식 GitHub](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) | 커밋 `81eed5e8f68b6bed1789f682fe78cdd324495afc`, MIT, `external/RVC` |
| FFmpeg·ffprobe | RVC 입력 디코딩 | [FFmpeg 공식 Windows 다운로드 안내](https://ffmpeg.org/download.html), [Gyan 빌드](https://www.gyan.dev/ffmpeg/builds/) | 8.1.2 essentials 정적 빌드, SHA-256 검증, GPLv3, `external/ffmpeg` |
| HuBERT·RMVPE·RVC 초기 가중치 | 로컬 특징 추출·학습 | [RVC에서 안내하는 Hugging Face 저장소](https://huggingface.co/lj1995/VoiceConversionWebUI) | 리비전 고정, SHA-256 검증, `external/RVC/assets` |
| Shotcut 번들의 whisper-cli | 보컬 한국어 발음의 참고용 자동 인식 | [Shotcut 다운로드](https://shotcut.org/download/), [whisper.cpp 코드](https://github.com/ggml-org/whisper.cpp) | **선택**, 이미 이 PC에 있던 `C:\Program Files\Shotcut\whisper-cli.exe`와 `share\shotcut\whisper_models\ggml-base-q5_1.bin` 사용. 앱 실행 의존성은 아니며 로컬 API 없음. Shotcut은 GPLv3, whisper.cpp 코드는 MIT |

## 원곡 분석·GUI·개사 추가 도구

| 도구 | 용도 | 공식 설치·저장소 | 버전·라이선스 | 필수 여부·경로·API |
| --- | --- | --- | --- | --- |
| PySide6 | Windows 데스크톱 GUI, 오디오 재생 | [Qt for Python](https://doc.qt.io/qtforpython-6/) / [GitHub](https://github.com/qtproject/pyside-pyside-setup) | 6.12.0, LGPLv3/GPLv3/상용 선택 | 필수, `.venv`; 외부 API 없음 |
| Demucs | 원곡 보컬/반주 분리 | [공식 GitHub](https://github.com/facebookresearch/demucs) / [PyPI](https://pypi.org/project/demucs/) | 4.0.1, MIT | 필수, `.venv-rvc`; 모델 `models/torch`, API 없음 |
| faster-whisper | 원곡 가사 인식 | [공식 GitHub](https://github.com/SYSTRAN/faster-whisper) / [PyPI](https://pypi.org/project/faster-whisper/) | 1.2.1, MIT | 필수, `.venv-rvc`; 모델 `models/faster-whisper-base`, API 없음 |
| Ollama | 로컬 LLM 자동 개사 | [공식 설치](https://ollama.com/download/windows) / [GitHub](https://github.com/ollama/ollama) | 사용자가 설치한 버전, MIT | **선택**, `ollama` 명령과 `127.0.0.1:11434/api/generate` |
| Git | RVC 코드 설치 | [공식 설치](https://git-scm.com/download/win) / [GitHub](https://github.com/git/git) | 설치 버전, GPLv2 | 최초 설치 시 필수, `git` 명령 |

GUI/SVS 패키지는 `requirements.txt`, 분석·RVC 패키지는 `requirements-rvc.txt`에 있습니다. 실제 설치한 전체 버전은 `requirements-lock-gui.txt`, `requirements-lock-rvc.txt`에 기록했습니다. RVC 추론은 `HF_HUB_OFFLINE=1`로 실행하고, Whisper는 로컬 경로로만 로드합니다.

모델 경로는 저장소 기준 `external/`, `models/` 아래입니다. 두 독립 가상환경을 `setup.bat`이 준비합니다. `external/ffmpeg`는 RVC·Demucs 실행 시 해당 프로세스의 PATH 앞에 추가합니다.
