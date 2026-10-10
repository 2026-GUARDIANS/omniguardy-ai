# Vision FastAPI 시연 API 명세서

이 문서는 현재 [`api/vision_api.py`](vision_api.py) 구현을 최종 기준으로 한 시연용 계약이다. 예전 `Vision responseDTO` 명세는 이 문서에 포함하지 않으며, 본 문서가 최신 계약이다.

## 실행 모델

시연은 Temporal Trial 6을 사용한다. API 서버 실행 환경에 다음 설정이 **필수**다.

```powershell
$env:VISION_MODEL = "temporal_trial6"
.\.venv\Scripts\python.exe api\vision_api.py
```

`VISION_MODEL`을 지정하지 않으면 구현의 기본값은 `xgboost`다. 따라서 환경 변수를 생략하면 Trial 6 시연 계약이 적용되지 않는다.

Trial 6의 내부 분류 클래스는 `N1`, `A17`, `A18_20`, `A19`, `A21`이다. API 경계에서는 내부 `A18_20`을 `A18`로 변환하며, `A20`은 독립 Trial 6 예측이 아니므로 API 확률이 항상 `0.0`이다.

## 요청

`POST /analyze/vision` (`multipart/form-data`)

| 필드 | 형식 | AUDIO | KEYPAD | 설명 |
|---|---|---|---|---|
| `file` | MP4 파일 | 필수 | 필수 | 최대 100 MB |
| `triggerId` | string | 필수 | 필수 | 빈 문자열 불가 |
| `triggerType` | `AUDIO` \| `KEYPAD` | `AUDIO` | `KEYPAD` | 필수 |
| `securityEventId` | string 또는 null | 선택 | 선택 | 공백만의 값, 대소문자 무시 `null`/`none`은 `null`로 정규화 |
| `triggeredAt` | ISO-8601 datetime | 필수 | 필수 | |

## 공통 성공 응답

```json
{
  "status": "success",
  "result": {
    "trigger": {
      "triggerId": "string",
      "triggerType": "AUDIO | KEYPAD",
      "securityEventId": "string or null",
      "triggeredAt": "ISO-8601 datetime"
    },
    "analyzedAt": "ISO-8601 datetime",
    "video": {
      "fileName": "string",
      "durationSeconds": 0.0,
      "fps": 0.0,
      "frameCount": 0
    },
    "behavior": "AUDIO object or null",
    "visionEvents": [],
    "observations": {
      "trackedPersonCount": 0,
      "hasTracking": false,
      "hasPose": false
    }
  }
}
```

- `video`와 `observations`는 모든 성공 응답에 포함된다.
- `securityEventId`는 KEYPAD에서 언제나 null로 강제되지 않는다. 요청으로 유효한 값이 전달되면 그 값이 응답에도 반환될 수 있다.
- Vision은 `riskLevel`이나 `riskScore`를 반환하지 않는다. Agent가 최종 위험도를 판단한다.

## AUDIO Trigger의 `behavior`

`behavior`는 AUDIO에서만 객체이며, KEYPAD에서는 `null`이다.

```json
{
  "prediction": "N1 | A17 | A18 | A19 | A20 | A21",
  "label": "string",
  "confidence": 0.0,
  "classProbabilities": {
    "N1": 0.0,
    "A17": 0.0,
    "A18": 0.0,
    "A19": 0.0,
    "A20": 0.0,
    "A21": 0.0
  }
}
```

Trial 6에서 `prediction: "A18"`의 실제 `label`은 `door_lock_suspicious_activity`다. 시연 시나리오 명칭은 **문 열기 시도**로 사용한다. Trial 6 API의 `A20` 확률은 항상 `0.0`이며, A20은 Trial 6의 독립 예측/Agent 매핑 대상이 아니다.

### Agent 시연 매핑

| API `behavior.prediction` | 시연 행동 의미 | Agent 처리 |
|---|---|---|
| `N1` | 정상 행동 | 위험 없음 |
| `A21` | 반복적인 문 두드림 | LOW |
| `A18` | 문 열기 시도 | MEDIUM |
| `A17` | 반복적인 초인종 누르기 | 로그만 기록 |
| `A19` | 문 발차기 | 로그만 기록 |
| `A20` | API 호환 값 | Trial 6 독립 매핑 없음 |

MEDIUM 판정은 AUDIO의 `behavior.prediction = "A18"`에 따른다. 키패드 입력 횟수·키패드 로그·별도 키패드 조작 추정은 이 Vision 계약의 MEDIUM 필수 조건이 아니다.

### AUDIO 예시 — LOW

```json
{
  "status": "success",
  "result": {
    "trigger": {"triggerId": "trg-audio-001", "triggerType": "AUDIO", "securityEventId": "sec-001", "triggeredAt": "2026-09-23T15:30:00+09:00"},
    "analyzedAt": "2026-09-23T15:30:10+09:00",
    "video": {"fileName": "door_video.mp4", "durationSeconds": 10.0, "fps": 30.0, "frameCount": 300},
    "behavior": {"prediction": "A21", "label": "long_or_repeated_door_knocking", "confidence": 0.94, "classProbabilities": {"N1": 0.03, "A17": 0.01, "A18": 0.02, "A19": 0.00, "A20": 0.0, "A21": 0.94}},
    "visionEvents": [],
    "observations": {"trackedPersonCount": 1, "hasTracking": true, "hasPose": true}
  }
}
```

### AUDIO 예시 — MEDIUM

```json
{
  "status": "success",
  "result": {
    "trigger": {"triggerId": "trg-audio-002", "triggerType": "AUDIO", "securityEventId": "sec-002", "triggeredAt": "2026-09-23T15:33:00+09:00"},
    "analyzedAt": "2026-09-23T15:33:10+09:00",
    "video": {"fileName": "door_video_02.mp4", "durationSeconds": 10.0, "fps": 30.0, "frameCount": 300},
    "behavior": {"prediction": "A18", "label": "door_lock_suspicious_activity", "confidence": 0.85, "classProbabilities": {"N1": 0.05, "A17": 0.03, "A18": 0.85, "A19": 0.01, "A20": 0.0, "A21": 0.06}},
    "visionEvents": [],
    "observations": {"trackedPersonCount": 1, "hasTracking": true, "hasPose": true}
  }
}
```

## KEYPAD Trigger의 `visionEvents`

KEYPAD는 `behavior: null`이며 규칙 기반 `visionEvents`를 반환한다. 감지 결과가 없으면 빈 배열이다.

기본 이벤트는 다음 두 종류다.

| eventType | 의미 | Agent 시연 처리 |
|---|---|---|
| `SILENT_SIGNAL` | MediaPipe 기반 비상 손동작 규칙 | HIGH |
| `REAR_CLOSE_APPROACH_SUSPECTED` | Person Tracking 기반 후방 근접·접근 규칙 | HIGH |

`VISION_ENABLE_WEAPON_DETECTION=true`이면 KEYPAD 분석에서 `WEAPON_PROXIMITY_SUSPECTED` 이벤트도 추가될 수 있다. 이 이벤트는 코드상 실험적 COCO knife/scissors 근접 탐지이며, 기본 이벤트나 기본 시연 위험도 계약에는 포함하지 않는다.

### KEYPAD 예시 — SILENT_SIGNAL

```json
{
  "status": "success",
  "result": {
    "trigger": {"triggerId": "trg-keypad-001", "triggerType": "KEYPAD", "securityEventId": null, "triggeredAt": "2026-09-23T15:31:00+09:00"},
    "analyzedAt": "2026-09-23T15:31:08+09:00",
    "video": {"fileName": "keypad_window.mp4", "durationSeconds": 8.0, "fps": 30.0, "frameCount": 240},
    "behavior": null,
    "visionEvents": [{"eventType": "SILENT_SIGNAL", "confidence": 0.94, "detectedFrame": 156, "details": {"signalCode": "S1", "signalName": "emergency_hand_gesture"}}],
    "observations": {"trackedPersonCount": 2, "hasTracking": true, "hasPose": true}
  }
}
```

## 백엔드 처리 규칙

```text
if triggerType == AUDIO:
    SecurityEvent에 behavior 결과를 저장·전달한다.
    Agent가 N1=정상, A21=LOW, A18=MEDIUM으로 판단한다.
    A17/A19는 로그만 기록한다.
    Trial 6의 A20은 독립 Agent 매핑 대상이 아니다.

if triggerType == KEYPAD:
    visionEvents가 비어 있으면 규칙 기반 이벤트 미감지로 처리한다.
    SILENT_SIGNAL 또는 REAR_CLOSE_APPROACH_SUSPECTED가 있으면
    Agent의 HIGH 시나리오 처리 대상으로 전달한다.
    WEAPON_PROXIMITY_SUSPECTED는 옵션 기능이 켜진 경우에만 발생할 수 있으며,
    별도 정책이 합의되기 전에는 기본 시연 매핑에 포함하지 않는다.
```

## 구현 정합성 점검 기준

- AUDIO/KEYPAD 분기, `behavior`의 null 규칙, `visionEvents` 배열, `observations`, 6개 확률 키, Trial 6의 A18 변환은 `vision_api.py` 구현과 일치한다.
- `VISION_KEYPAD_HAND_RULE`은 내부 진단 로그용이며, `KEYPAD_REPEATED_HAND_APPROACH`를 API 응답 `visionEvents`에 추가하지 않는다.
