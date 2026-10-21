# Entrance Behavior Vision Pipeline

현관 카메라 영상에서 정상 활동과 이상행동을 블록 단위로 분류하는 Vision 파이프라인입니다.

## 분류 라벨

| 라벨 | 의미 |
|---|---|
| N1 | 정상 현관 활동: 거주자 출입, 일반 배달, 대기, 이탈 |
| A17 | 초인종 반복 누름 |
| A18 | 문 열기 반복 시도 |
| A19 | 문 반복 발차기 |
| A20 | 문 안쪽을 들여다보기 시도 |
| A21 | 장시간 또는 반복 문 두드림 |

## 데이터 분할

| 구분 | 사용 데이터 |
|---|---|
| Train | AIHub TL/TS, mydata, train 원본에서 파생된 augmentation |
| Validation | AIHub VL/VS 원본 영상만 사용 |
| Test | 별도 전시 카메라 또는 보류 현장 영상으로 추후 구성 |

mydata의 `normal`, `delivery`는 N1으로 사용하고, `lookingInside`는 A20으로 사용합니다.

## 처리 흐름

```text
영상
→ YOLO 사람 탐지 + BoT-SORT tracking
→ MediaPipe Pose 추출
→ annotation 블록별 행동·pose 특징 집계
→ XGBoost 분류
→ N1 / A17~A21 예측 결과
```

- tracking과 pose는 영상당 한 번만 실행합니다.
- 특징과 학습은 annotation의 `start_frame`, `end_frame` 블록 범위를 사용합니다.
- train의 augmentation은 원본 블록 경계를 그대로 사용합니다.

## 실행 순서

가상환경을 활성화한 뒤 프로젝트 루트에서 실행합니다.

```powershell
python .\utils\load_annotation.py
python .\utils\generate_annotations.py
python .\tracking\tracker.py
python .\tracking\extract_behavior_features.py
python .\tracking\pose_extractor.py
python .\tracking\merge_features.py
python .\models\train_XGBoost.py
```

`tracker.py`와 `pose_extractor.py`는 영상별 CSV가 이미 있으면 건너뛰므로, 중단 후 재실행할 수 있습니다.

## 생성 결과

| 경로 | 내용 |
|---|---|
| `splits/annotations/train_annotations.csv` | TL/TS 기반 AIHub train 블록 manifest |
| `splits/annotations/valid_annotations.csv` | VL/VS 기반 validation 블록 manifest |
| `splits/annotations/train_annotations_all.csv` | AIHub + mydata + augmentation train manifest |
| `results/tracking/` | 영상별 사람 tracking 결과 |
| `results/pose/` | 영상별 pose 프레임 특징 |
| `results/features/` | 블록별 behavior·pose 병합 특징 |
| `results/models/xgboost_validation/` | validation report, confusion matrix, 모델 파일 |

## 현재 baseline 결과

AIHub VL/VS hold-out validation 기준:

- Accuracy: **86.36%**
- Macro F1-score: **68.82%**
- Weighted F1-score: **86.20%**

N1 비중이 높으므로 Accuracy만이 아니라 Macro F1-score를 함께 평가합니다. 이 결과는 validation 성능이며, 최종 일반화 성능은 전시 카메라 또는 별도 현장 test 영상으로 추가 검증해야 합니다.

## 전시 Vision FastAPI

`POST /analyze/vision`은 `multipart/form-data`로 MP4와 트리거 메타데이터를 받습니다. Vision은 관찰 이벤트 또는 기존 행동 분류 결과만 반환하며, LOW/MEDIUM/HIGH 위험도 결정은 Agent의 역할입니다.

| `triggerType` | 기본 처리 |
|---|---|
| `AUDIO` | `exhibition_rules`: 사람 누적 관찰 및 카메라 가림 규칙 |
| `KEYPAD` | 기존 `SILENT_SIGNAL`, `REAR_CLOSE_APPROACH_SUSPECTED` 규칙 |

`VISION_AUDIO_MODE`는 AUDIO 경로에만 적용됩니다. 미설정 또는 `exhibition_rules`면 전시 규칙을 실행하고, `temporal_trial6`을 명시하면 보존된 Temporal Trial 6 행동 분류 모델을 실행합니다. KEYPAD 경로는 이 설정과 무관하게 기존 규칙을 유지합니다.

```powershell
# 전시 기본 모드 (VISION_AUDIO_MODE 미설정과 동일)
$env:VISION_AUDIO_MODE = "exhibition_rules"
.\.venv\Scripts\python.exe .\api\vision_api.py

# 기존 Trial 6 백업 모드
$env:VISION_AUDIO_MODE = "temporal_trial6"
$env:VISION_MODEL = "temporal_trial6"
.\.venv\Scripts\python.exe .\api\vision_api.py
```

전시 규칙 AUDIO 응답은 `behavior: null`이며, `visionEvents`에 다음 관찰 이벤트를 반환합니다.

| 이벤트 | 판정 개요 |
|---|---|
| `PERSON_DWELL_OBSERVED` | 같은 BoT-SORT Track ID가 누적 5초 이상 관찰됨 |
| `CAMERA_OCCLUSION_SUSPECTED` | 첫 10초 안에서 저 edge density·저 Laplacian 상태가 0.3초 이상 지속됨 |

이 모드의 `observations`에는 선택 필드 `analysisMode: "exhibition_rules"`가 포함됩니다. `temporal_trial6`과 KEYPAD 기존 응답에는 이 필드를 추가하지 않습니다. 상세 API 요청·응답 계약은 `api/vision_api.py` 구현을 기준으로 관리합니다.

## Git 관리

원본·증강 영상, 실행 결과, 자동 생성 manifest, 모델 가중치는 `.gitignore`로 제외합니다. 코드와 재현에 필요한 설정 파일만 Git으로 관리합니다.
