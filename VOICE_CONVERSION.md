# 2단계: 허락받은 말하기 음성으로 노래 음색 변환

이 절차는 [RVC 공식 저장소](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)의 코드를 로컬에서 실행한 초기 기술 검증 기록입니다. 현재 GUI는 별도의 `scripts/train_voice_profile.py`로 이 학습 과정을 자동화합니다. 원본 음성 제공자가 **AI 학습과 음색 변환을 명시적으로 허락한 경우에만** 진행하세요. 학습 데이터, 체크포인트, 출력 WAV는 Git에서 제외됩니다.

## 설치

저장소 루트에서 Python 3.10용 `.venv`(한국어 SVS용)를 먼저 만들고 [INSTALL.md](INSTALL.md)를 실행합니다. 별도로 Python 3.12 x64를 설치한 후 다음을 실행합니다.

```powershell
py -3.12 -m venv .venv-rvc
.\.venv-rvc\Scripts\python.exe -m pip install torch==2.7.1+cu118 torchaudio==2.7.1+cu118 --index-url https://download.pytorch.org/whl/cu118
.\.venv-rvc\Scripts\python.exe -m pip install -r requirements-rvc-poc.txt
.\.venv\Scripts\python.exe -m pip install "huggingface_hub>=0.35,<1"
git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git external\RVC
git -C external\RVC checkout 81eed5e8f68b6bed1789f682fe78cdd324495afc
.\.venv\Scripts\python.exe scripts\fetch_rvc_assets.py
.\.venv\Scripts\python.exe scripts\fetch_ffmpeg.py
```

RVC 모델 파일은 Hugging Face 리비전 `e6d0c1a17da07c33557852f9dfa2bd44cc75737d`로 고정하고 SHA-256을 검사합니다. FFmpeg 8.1.2 Windows essentials ZIP도 공식 배포 체크섬을 검사합니다. 설치 후 아래 학습·추론 과정에서 Hugging Face는 오프라인 모드로 설정합니다. Python 패키지와 모델을 받을 때만 인터넷이 필요합니다.

## 데이터 선택과 학습

아래 예시는 **이미 AI 음성 학습·변환에 사용 허락받은** `C:\voice_samples`를 사용합니다. 자신의 허락받은 음성 폴더 경로로 바꾸세요. 선택 스크립트는 입력 WAV를 읽기만 하고, 길이·레벨·파일명의 잡음 표기를 기준으로 별도 폴더에 복사합니다. 이 규칙은 청취 품질 심사를 대체하지 않습니다.

```powershell
.\.venv\Scripts\python.exe scripts\prepare_rvc_voice.py 'C:\voice_samples' voice_data\rvc_proof --seconds 900
Set-Location external\RVC
$env:PATH = (Resolve-Path '..\ffmpeg').Path + ';' + $env:PATH
$env:CUDA_VISIBLE_DEVICES = '0'
$env:HF_HUB_OFFLINE = '1'
$RVC_PY = (Resolve-Path '..\..\.venv-rvc\Scripts\python.exe').Path
$InputClips = (Resolve-Path '..\..\voice_data\rvc_proof\clips').Path
New-Item -ItemType Directory -Force logs\rvc_proof | Out-Null
& $RVC_PY -m train.preprocess $InputClips 32000 8 logs\rvc_proof False 3.7
& $RVC_PY -m train.dataset.extract_f0 cuda 1 0 0 logs\rvc_proof false
& $RVC_PY -m train.dataset.extract_hubert_feature cuda:0 1 0 0 logs\rvc_proof v2 false
& $RVC_PY ..\..\scripts\prepare_rvc_manifest.py
& $RVC_PY -m train.train_index rvc_proof v2 assets\indices 8 single
& $RVC_PY -m train.train -e rvc_proof -sr 32k -f0 1 -bs 2 -g 0 -te 10 -se 5 -pg assets\pretrained_v2\f0G32k.pth -pd assets\pretrained_v2\f0D32k.pth -l 1 -c 0 -sw 1 -v v2
```

RVC의 `train/preprocess.py`를 Windows에서 파일 경로로 직접 실행하면 `train.py`가 `train` 패키지를 가리는 문제가 생겨 여기서는 `-m train.preprocess`를 사용합니다. GTX 1660 SUPER와 GTX 1060은 이 RVC 코드에서 FP32로 선택됩니다. GPU 번호는 시스템마다 다르며, 이 PC의 1660 SUPER는 0번입니다. `-bs 2`는 6GB 메모리 범위에서 실행한 시험 설정입니다.

## 변환 및 검사

학습이 끝나면 `assets\weights\rvc_proof_e10_*.pth`와 `assets\indices\*rvc_proof*.index`가 생깁니다. 1단계 SVS WAV에 적용합니다.

```powershell
$Model = Get-ChildItem assets\weights -Filter 'rvc_proof_e10_*.pth' | Select-Object -First 1
$Index = Get-ChildItem assets\indices -Filter '*rvc_proof*.index' | Select-Object -First 1
$InputWav = (Resolve-Path '..\..\outputs\proof_korean_svs.wav').Path
$OutputWav = Join-Path (Resolve-Path '..\..\outputs').Path 'proof_voice_converted.wav'
& $RVC_PY -m infer.cli --model $Model.FullName --input $InputWav --output $OutputWav --pitch 0 --f0-method rmvpe --index $Index.FullName --index-rate 0.75
Set-Location ..\..
.\.venv\Scripts\python.exe scripts\analyze_voice_conversion.py outputs\proof_korean_svs.wav outputs\proof_voice_converted.wav --json outputs\proof_voice_conversion.json
```

길이, RMS, 클리핑, 음정 차이는 자동 측정합니다. 화자 유사도, 한국어 자음 보존, 고음의 자연스러움은 실제 청취 평가가 필요합니다. 10에포크는 기술 검증용 짧은 학습이며 완성 음질 기준이 아닙니다.

더 강한 오프라인 확인에는 위 변환 명령의 `-m infer.cli` 대신 `..\..\scripts\prove_rvc_offline.py`를 넣어 실행합니다. 이 래퍼는 Python 소켓 생성을 차단하며, 이 PC에서 최종 WAV가 정상 생성됐습니다.
