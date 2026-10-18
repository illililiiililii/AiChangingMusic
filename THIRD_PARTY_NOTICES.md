# 제3자 고지

이 저장소는 모델 가중치나 제3자 코드를 기본적으로 재배포하지 않습니다. `scripts/fetch_diffsingerkr.py`는 사용자의 PC에 CODEJIN의 [DiffSingerKR Space](https://huggingface.co/spaces/codejin/diffsingerkr) 파일을 다운로드합니다. 코드 저장소의 [MIT 라이선스](https://github.com/CODEJIN/DiffSingerKR/blob/main/LICENSE)와 Space의 MIT 메타데이터를 확인했습니다. 그러나 체크포인트의 학습 데이터 [CSD는 CC BY-NC-SA 4.0](https://github.com/equal-singer/CSD#6-license)이고 상업적 사용을 금지합니다. 따라서 이 모델의 상업적 사용 권리를 확인했다고 주장하지 않습니다. 다운로드된 코드의 사용·재배포 시 원저작자의 저작권 및 라이선스 고지를 유지해야 합니다.

2단계에서는 [RVC 공식 코드를 MIT 라이선스](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/LICENSE)로 로컬에 복제하고, [RVC에서 안내하는 모델 저장소](https://huggingface.co/lj1995/VoiceConversionWebUI)의 HuBERT·RMVPE·32k 초기 가중치를 직접 다운로드합니다. 해당 Hugging Face 저장소는 MIT로 표시되지만 모델 카드가 비어 있어 학습 데이터 출처와 권리 범위에 대한 추가 확인이 필요합니다. 이 프로젝트는 해당 파일이나 사용자 학습 체크포인트를 Git에 포함하지 않습니다.

Windows FFmpeg 바이너리는 [FFmpeg가 연결하는 Gyan의 Essentials 빌드](https://www.gyan.dev/ffmpeg/builds/)를 SHA-256 확인 후 로컬에 둡니다. 배포자가 명시한 바이너리 라이선스는 GPLv3입니다. 바이너리를 이 저장소에 재배포하지 않습니다.

PyTorch 및 Python 패키지에는 각각의 라이선스가 적용됩니다. 직접 설치한 패키지와 버전은 `requirements.txt`, `requirements-rvc.txt`, 각 lock 파일을 참조합니다. [Demucs](https://github.com/facebookresearch/demucs)는 MIT 라이선스 코드, [faster-whisper](https://github.com/SYSTRAN/faster-whisper)는 MIT 라이선스 코드입니다. 두 모델의 가중치는 각 공식 다운로드처에서 사용자 PC로 받으며 이 Git 저장소에 재배포하지 않습니다. [PySide6](https://doc.qt.io/qtforpython-6/)는 LGPLv3/GPLv3/상용 라이선스 선택지에 따라 사용합니다.
발음 참고 평가에는 이 PC에 기존 설치된 Shotcut의 `whisper-cli.exe`와 한국어 지원 Whisper base 모델을 사용했습니다. [Shotcut은 GPLv3](https://shotcut.org/FAQ/), [whisper.cpp 코드는 MIT](https://github.com/ggml-org/whisper.cpp)입니다. 이 도구와 모델은 프로젝트에 포함하거나 설치 요구사항으로 강제하지 않습니다.

입력한 음악, 가사, 목소리의 권리는 모델 코드의 오픈소스 라이선스가 대신 허가하지 않습니다. 본인 또는 명시적으로 허락받은 목소리만 사용해야 합니다.
