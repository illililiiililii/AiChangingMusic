# 1단계 한국어 노래 합성 기술 검증 기록

조사 기준일: 2026-10-18. 아래에서 **문서 확인**, **로컬 실행 확인**, **미검증**을 구분합니다. 공개 데모가 작동한다는 사실만으로 이 PC의 6GB GPU 실행이나 목표 화자 음색을 보증하지 않습니다.

## 노래 합성 후보

| 기술 | 한국어 입력·가중치 | 제어 | 코드 라이선스 | 판단 |
| --- | --- | --- | --- | --- |
| [CODEJIN/DiffSingerKR](https://github.com/CODEJIN/DiffSingerKR) | 한글 자소 기반 구현과 [683MB 공개 체크포인트](https://huggingface.co/spaces/codejin/diffsingerkr/blob/main/Checkpoint/S_200000.pt) 확인. Space 예제의 음역은 MIDI 65~89 | 음절, 음표, 길이 직접 입력 | 저장소 MIT, [Space 메타데이터 MIT](https://huggingface.co/spaces/codejin/diffsingerkr/blob/main/README.md). **학습 데이터 CSD는 CC BY-NC-SA 4.0** | 비상업적 첫 검증용으로 선정. 22.05kHz, 어린이 노래 데이터의 음색·음역 한계 가능. 오래된 코드라 Windows 의존성 패치 필요 |
| [MLP Singer](https://github.com/neosapience/mlp-singer) | 한국어 체크포인트와 HiFi-GAN 보코더 안내 | 가사·MIDI·길이 데이터 | 코드 MIT, 체크포인트 학습 데이터 CSD는 CC BY-NC-SA 4.0 | 대안 후보. 보코더 별도 다운로드, 예제 중심 CLI. 이번 PC 실행은 미검증 |
| [OpenVPI DiffSinger](https://github.com/openvpi/DiffSinger) | 한국어용 공식 체크포인트와 음소 사전은 이번 조사에서 확인하지 못함 | 세밀한 음정·에너지·호흡 제어 | Apache-2.0 | 이후 더 나은 SVS 엔진 후보. 한국어 모델과 가중치 권리 확보가 선행되어야 함 |
| [원본 DiffSinger](https://github.com/MoonInTheRiver/DiffSinger) | 공개 예제는 중국어 PopCS 중심. 한국어 가중치 확인 못함 | MIDI 및 F0 조건 | MIT | 한국어 MVP에 바로 채택하지 않음 |

선정 근거: 한국어 문자를 실제로 처리하는 공개 체크포인트와 명시적 음표·시간 입력을 동시에 찾은 후보는 CODEJIN 모델입니다. 현재의 PyTorch 호환성, 6GB VRAM 사용량, 발음·피치 정확도는 반드시 로컬 결과로 판단합니다. [원본 논문](https://arxiv.org/abs/2105.02446)은 DiffSinger가 음표 기반 SVS임을 설명합니다.

라이선스 판단: CODEJIN은 CSD로 체크포인트를 훈련했다고 명시하며, [CSD 제공자는 비상업적 이용만 허용](https://github.com/equal-singer/CSD#6-license)합니다. Space의 `license: mit`는 이 데이터 사용 제한을 해소한다고 확인할 수 없습니다. 이 체크포인트를 상업적 사용 가능 모델로 소개하거나 저장소에 재배포하지 않습니다. 상업 배포에는 별도 권리 확인 또는 권리가 명확한 데이터로 학습한 대체 체크포인트가 필요합니다.

## 말하기 음성 → 노래 음색 변환 후보

| 기술 | 말하기 데이터만 사용 | GPU·Windows | 라이선스와 상태 | 판단 |
| --- | --- | --- | --- | --- |
| [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) | 저장소는 저잡음 음성 **최소 10분 권장**, 사전학습 가중치로 학습하며 노래용으로 F0 모델 선택을 안내 | Windows, GTX 1660 SUPER 6GB에서 학습·변환 실행 확인 | 코드 MIT. 가중치·사용자 음성 권리는 따로 점검 | 2단계에서 실제 사용. 말하기로 학습한 음역 밖 노래에서 품질 하락 가능 |
| [Applio](https://github.com/IAHispano/Applio) | RVC 계열 학습·변환 도구 | Windows 설치 스크립트 제공. 6GB 성공 미검증 | 저장소 MIT 및 별도 [이용약관](https://github.com/IAHispano/Applio/blob/main/TERMS_OF_USE.md) | RVC 자체의 설치·운영 대안. 약관 검토 후 선택 |
| [so-vits-svc](https://github.com/svc-develop-team/so-vits-svc) | 음성 특징과 F0로 변환. 말하기 학습으로 노래 품질은 미검증 | Windows 요구사항 있으나 유지보수 위험 | AGPL-3.0, 2023년 저장소 보관 | 첫 MVP에서는 제외 |

사용자가 명시적 허락을 확인한 말하기 음성으로 2단계 RVC 학습과 11.16초 변환 WAV 생성을 완료했습니다. 실제 입력, GPU 메모리, 오프라인 검증, 발음 인식의 한계는 [VOICE_CONVERSION_VALIDATION.md](VOICE_CONVERSION_VALIDATION.md)에 기록했습니다. 화자 유사도·고음 자연스러움·한국어 명료도는 아직 청취 검증이 필요합니다.

현 RVC 설정 코드는 [GTX 16 계열과 SM 6.1 GPU를 FP32로 실행](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/configs/config.py)합니다. 따라서 1660 SUPER와 1060에서 FP16 절감을 전제로 VRAM을 계산하면 안 됩니다. 2단계 학습·추론 중 `nvidia-smi` 관측 최고치는 각각 약 5,158MiB·2,992MiB이며 정확한 피크 계측은 아닙니다.

## 하드웨어 및 로컬 실험

- Windows에서 `Win32_VideoController`로 GTX 1060 6GB와 GTX 1660 SUPER를 확인했습니다. 이 WMI 속성의 `AdapterRAM` 값은 4GB로 잘려 표시되므로 **VRAM 실측으로 사용하지 않습니다**. PyTorch CUDA는 1660 SUPER를 `cuda:0`(6,441,992,192 bytes, SM 7.5), 1060을 `cuda:1`(6,442,319,872 bytes, SM 6.1)로 확인했습니다.
- 시스템 Python 3.10.6은 설치되어 있었지만 PATH에 없었습니다. 별도 `.venv`를 만들었습니다. FFmpeg와 `nvidia-smi`는 PATH에서 찾지 못했습니다.
- 모델은 Hugging Face 리비전 `26f95ac2554b6a6725706ed2a7991a62f7f714a9`로 고정했습니다. 체크포인트 SHA-256 `6482992a43b8a98554e7ef9e487a381c2717c5828d564e6dfc6cac16a0e16092`, 보코더 SHA-256 `5b47a5d03d744861f94ee973294317f738ccc6dc6d27bafa5d8db5ed18f95566` 확인.
- 공개 Space의 추론 코드가 사용하지 않는 `pysptk`를 모듈 import 때 요구합니다. 다운로드 스크립트가 그 import 한 줄만 제거합니다. 원본 Space와 프로젝트 코드는 Git에 포함하지 않습니다.
- **로컬 WAV 생성 확인:** `outputs/proof_korean_svs.wav` (mono 22,050Hz PCM 16bit, 246,272프레임, 11.169초). 25개 새 한국어 음절을 지정 시간과 멜로디로 생성했습니다. 길이 오차 -0.001초, RMS 약 0.229, 피크 약 0.97, 클리핑 샘플 0개입니다.
- **음정 확인:** 공개 체크포인트는 입력 음표 ID보다 출력 F0가 약 **한 옥타브 낮았습니다**. 이 모델 전용 어댑터에서 목표 MIDI 음정에 12를 더해 입력합니다. 보정 후 `librosa.pyin`의 음표 중간 구간 F0가 25/25음표에서 검출됐고 절대 오차 중앙값 10.6센트, 100센트 이내 25/25였습니다. 이는 F0 자동 측정치이며 발음 품질을 증명하지는 않습니다.
- **GPU 확인:** GTX 1660 SUPER `cuda:0`에서 CUDA OOM 없이 추론했습니다. PyTorch 피크 할당 471.8MiB, 예약 558MiB, 추론 중 CUDA 여유 메모리 감소 약 1,339MiB(컨텍스트 등 포함)입니다. 체크포인트 로드를 포함한 스크립트 내부 시간은 약 3.6초였습니다. GTX 1060 6GB `cuda:1`도 같은 구간을 생성했습니다(피크 할당 740.9MiB, 예약 1,144MiB, 여유 메모리 감소 약 2,268MiB, 내부 시간 약 4.0초). 이 값은 한 번의 짧은 구간 실측으로, 장시간 곡 처리의 상한은 아닙니다.
- **오프라인 추론 확인:** 모델 파일 준비 후 Python 소켓 생성을 차단한 상태에서 같은 WAV를 다시 생성했습니다. 발음의 자연스러움은 사람이 실제로 듣고 평가해야 하므로 **미검증**입니다. 목표 목소리 변환은 2단계에서 실제 실행했으며 별도 기록에 있습니다.

## 2단계 연결 검증과 남은 품질 기준

1. 10~20초의 새 한국어 문구를 지정 MIDI 음표와 길이로 6GB GPU에서 OOM 없이 생성.
2. 출력 WAV가 비어 있지 않고 길이 오차가 0.2초 이하.
3. 음절별 중심부 F0 오차와 무음 비율을 수치화하고, 한국어 발음은 실제 청취로 판정.
4. 허가된 말하기 음성 최소 10분으로 RVC 프로필을 학습한 뒤 **같은 SVS WAV**를 변환: 2단계에서 15분 입력·11.16초 출력을 확인. 음색 유사도, 고음 안정성, 자음 유지는 청취 평가가 필요합니다.
5. 오프라인 재실행은 확인했습니다. 관측 GPU 사용량은 기록했지만 계측 범위상 정확한 피크로 간주하지 않습니다. 두 GPU 메모리는 합산하지 않습니다.
