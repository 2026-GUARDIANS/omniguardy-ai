# Streaming Vision API — Spring Boot JSON 계약 (실제 구현 기준)

작성일: 2026-11-02
대상 서버: `experiments/streaming_poc/streaming_api_poc/app.py`
범위: 독립 Streaming PoC 포트 8010. 보호된 MP4 API(`api.vision_api`, 포트 8000)는 이 문서의 대상이 아니다.

아래 JSON은 localhost FastAPI 호출에서 수집한 원문을 우선 사용했다. `sessionId`, UUID, 시간, callback URL만 실행마다 달라진다. 문서와 구현이 다르면 **구현이 우선**한다.

## 공통 타입 및 상태값

| JSON 위치 | 타입 | 값 / null 규칙 |
|---|---|---|
| `analysisState` | string | lifecycle: `RECEIVING`, `PROCESSING`, `COMPLETED`, `FAILED` |
| `analysisCompleteness` | string | coverage: `PROCESSING`, `COMPLETE_KEYPAD`, `COMPLETE_AUDIO`, `PARTIAL_SAMPLED_TIME_AXIS`, `FAILED` |
| `status` | string | FINAL_RESULT: `success`, `partial`, `failed`; start: `started`; frame: `accepted`/`duplicate`; complete: `input_complete` |
| `result.trackedPersonCount` | integer \| null | final tracking 고유 ID 수. complete analysis에서 `0`도 유효; partial/failed에서는 `null` 가능 |
| `result.visionEvents` | array<object> | 빈 배열 가능. 이벤트 object 스키마는 eventType별로 다름 |
| `error` | string \| null | worker/분석 오류. 정상/partial 성공 경로에서는 null |
| timestamp | string \| null | `capturedAt`은 입력에서 optional이고 형식 검증 없음. `emittedAt`은 ISO-8601 timezone string |

`analysisState=COMPLETED`가 곧 완전 분석을 뜻하지는 않는다. `analysisCompleteness=PARTIAL_SAMPLED_TIME_AXIS`이면 `status=partial`이며 MP4 fallback 대상이다.

## 1. `POST /vision/start`

Content-Type: `application/json`

### 요청 필드

| 필드 | JSON 타입 | 필수 | nullable | 구현 규칙 |
|---|---|---:|---:|---|
| `sessionId` | string | 예 | 아니오 | 1~128자. Pi가 생성. 이미 존재하면 409 |
| `triggerId` | string | 조건부 | 예 | 신규 호출의 필수 식별자. `eventId` 중 하나는 반드시 필요 |
| `eventId` | string | 선택 | 예 | legacy alias. `triggerId`가 우선 |
| `securityEventId` | string | AUDIO면 예 | 예 | KEYPAD에서는 null/생략 가능. AUDIO면 422 |
| `triggerType` | string | 선택 | 아니오 | 기본 `KEYPAD`. 현재 구현은 문자열 자체를 Pydantic에서 제한하지 않음 |
| `sourceFps` | number | 예 | 아니오 | `0 < sourceFps <= 120` |
| `callbackUrl` | string | 선택 | 예 | 생략/null 가능. `http`/`https`와 서버 allowlist host만 허용 |
| `finalResultCallbackEnabled` | boolean | 선택 | 아니오 | 기본 false. Spring Boot 계약은 polling이므로 false 권장 |

수집 요청:

```json
{
  "sessionId": "spec-60bd4137",
  "triggerId": "spec-trigger",
  "triggerType": "KEYPAD",
  "sourceFps": 30.0
}
```

수집 201 응답:

```json
{
  "sessionId": "spec-60bd4137",
  "triggerId": "spec-trigger",
  "status": "started",
  "nextExpectedFrameIndex": 0,
  "scope": "online Silent Signal EARLY_EVENT PoC"
}
```

`scope`은 현재 구현 문자열이며 전체 MP4 호환성의 보증 필드가 아니다.

## 2. `POST /vision/frame`

Content-Type: `multipart/form-data` — JSON body가 아니다.

| multipart part | 타입 | 필수 | nullable | 구현 규칙 |
|---|---|---:|---:|---|
| `sessionId` | text string | 예 | 아니오 | start에서 만든 session |
| `frameIndex` | text → integer | 예 | 아니오 | 전송 순번, 반드시 0부터 연속 |
| `sourceFrameIndex` | text → integer | 선택 | 예 | 원본 촬영 프레임. 생략하면 `frameIndex` 사용 |
| `capturedAt` | text string | 선택 | 예 | 원본 촬영시각 전달용. 현재 형식 검증 없음 |
| `frame` | file | 예 | 아니오 | `image/jpeg` 또는 `image/jpg`만 허용 |

수집 202 accepted 응답:

```json
{
  "sessionId": "spec-60bd4137",
  "frameIndex": 0,
  "sourceFrameIndex": 0,
  "status": "accepted"
}
```

수집 202 duplicate 응답:

```json
{
  "sessionId": "spec-60bd4137",
  "frameIndex": 0,
  "sourceFrameIndex": 0,
  "status": "duplicate"
}
```

`frameIndex`는 sender 멱등성/순서용, `sourceFrameIndex`와 `capturedAt`은 원본 시간축용이다. 30 FPS 최종 분석에서는 둘이 보통 동일하게 증가한다.

## 3. `POST /vision/complete`

Content-Type: `application/json`

| 필드 | 타입 | 필수 | nullable |
|---|---|---:|---:|
| `sessionId` | string | 예 | 아니오 |

수집 요청과 202 응답:

```json
{"sessionId":"spec-60bd4137"}
```

```json
{
  "sessionId": "spec-60bd4137",
  "status": "input_complete",
  "resultUrl": "/vision/result/spec-60bd4137"
}
```

중복 complete도 202이다. 이 202은 input closure일 뿐 FINAL_RESULT 완료가 아니다.

## 4. `EARLY_EVENT` callback

callback URL은 start에서만 설정한다. callback transport는 비동기이고, 실패 시 최대 2회 시도한다. 같은 재시도는 같은 `eventInstanceId`, `sequence`, payload를 사용한다.

수집 원문(첫 callback 500 후 같은 payload로 204 재시도):

```json
{
  "messageType": "EARLY_EVENT",
  "sessionId": "contract-main-24720d91",
  "triggerId": "contract-main-trigger",
  "securityEventId": null,
  "triggerType": "KEYPAD",
  "eventInstanceId": "contract-main-24720d91-signal-001",
  "sequence": 1,
  "provisional": true,
  "event": {
    "eventType": "SILENT_SIGNAL",
    "detectedFrame": 114,
    "confirmedFrame": 114,
    "capturedAt": "3.800000"
  },
  "emittedAt": "2026-11-02T14:44:19.167+09:00"
}
```

| 위치 | 타입 | nullable |
|---|---|---:|
| `messageType` | string (`EARLY_EVENT`) | 아니오 |
| `sessionId`, `triggerId`, `triggerType`, `eventInstanceId` | string | 아니오 |
| `securityEventId` | string \| null | 예 |
| `sequence` | integer | 아니오 |
| `provisional` | boolean | 아니오 |
| `event.eventType` | string (`SILENT_SIGNAL`) | 아니오 |
| `event.detectedFrame`, `event.confirmedFrame` | integer | 아니오 |
| `event.capturedAt` | string \| null | 예 |
| `emittedAt` | string | 아니오 |

## 5. `GET /vision/status/{sessionId}`

모든 status 응답의 공통 필드 및 타입:

| 필드 | 타입 | nullable |
|---|---|---:|
| `sessionId`, `triggerId`, `triggerType`, `analysisState`, `analysisCompleteness` | string | 아니오 |
| `securityEventId`, `error` | string \| null | 예 |
| `inputComplete`, `finalResultAvailable` | boolean | 아니오 |
| `nextExpectedFrameIndex`, `queueDepth`, `mediaPipeQueueMaxDepth`, `yoloQueueMaxDepth`, `audioQueueMaxDepth`, `earlyEventCount`, `trackingRecordCount` | integer | 아니오 |
| `completedSessionTtlSecondsRemaining` | number \| null | 예 |
| `callbackAttempts` | array<object> | 아니오 |

`callbackAttempts[]`는 `messageType:string`, `eventInstanceId:string|null`, `attempt:integer`, `status:integer|null`, `error:string|null`, `at:string`, `elapsedMs:number`를 가진다.

### RECEIVING — 수집 원문

```json
{
  "sessionId": "spec-60bd4137",
  "triggerId": "spec-trigger",
  "securityEventId": null,
  "triggerType": "KEYPAD",
  "analysisState": "RECEIVING",
  "inputComplete": false,
  "nextExpectedFrameIndex": 0,
  "queueDepth": 0,
  "mediaPipeQueueMaxDepth": 0,
  "yoloQueueMaxDepth": 0,
  "finalResultAvailable": false,
  "completedSessionTtlSecondsRemaining": null,
  "earlyEventCount": 0,
  "callbackAttempts": [],
  "error": null,
  "analysisCompleteness": "PROCESSING",
  "audioQueueMaxDepth": 0,
  "trackingRecordCount": 0
}
```

주의: 구현상 RECEIVING 상태에서도 `analysisCompleteness` 초기값은 `PROCESSING`이다.

### PROCESSING — 수집 원문

```json
{
  "sessionId": "spec-60bd4137",
  "triggerId": "spec-trigger",
  "securityEventId": null,
  "triggerType": "KEYPAD",
  "analysisState": "PROCESSING",
  "inputComplete": true,
  "nextExpectedFrameIndex": 1,
  "queueDepth": 1,
  "mediaPipeQueueMaxDepth": 1,
  "yoloQueueMaxDepth": 1,
  "finalResultAvailable": false,
  "completedSessionTtlSecondsRemaining": null,
  "earlyEventCount": 0,
  "callbackAttempts": [],
  "error": null,
  "analysisCompleteness": "PROCESSING",
  "audioQueueMaxDepth": 0,
  "trackingRecordCount": 0
}
```

### COMPLETED — 수집 원문

```json
{
  "sessionId": "spec-60bd4137",
  "triggerId": "spec-trigger",
  "securityEventId": null,
  "triggerType": "KEYPAD",
  "analysisState": "COMPLETED",
  "inputComplete": true,
  "nextExpectedFrameIndex": 1,
  "queueDepth": 0,
  "mediaPipeQueueMaxDepth": 1,
  "yoloQueueMaxDepth": 1,
  "finalResultAvailable": true,
  "completedSessionTtlSecondsRemaining": 3599.957,
  "earlyEventCount": 0,
  "callbackAttempts": [],
  "error": null,
  "analysisCompleteness": "COMPLETE_KEYPAD",
  "audioQueueMaxDepth": 0,
  "trackingRecordCount": 0
}
```

### FAILED — 수집 원문

```json
{
  "sessionId": "failed-49051231",
  "triggerId": "failed-trigger",
  "securityEventId": null,
  "triggerType": "INVALID",
  "analysisState": "FAILED",
  "inputComplete": true,
  "nextExpectedFrameIndex": 1,
  "queueDepth": 0,
  "mediaPipeQueueMaxDepth": 1,
  "yoloQueueMaxDepth": 1,
  "finalResultAvailable": true,
  "completedSessionTtlSecondsRemaining": 3599.938,
  "earlyEventCount": 0,
  "callbackAttempts": [],
  "error": "unsupported triggerType=INVALID",
  "analysisCompleteness": "FAILED",
  "audioQueueMaxDepth": 0,
  "trackingRecordCount": 0
}
```

## 6. `GET /vision/result/{sessionId}`

호출 중이면 HTTP 202 (`{"detail":"analysis is still draining"}`)이다. terminal session이면 HTTP 200과 FINAL_RESULT object를 반환한다. FINAL_RESULT는 terminal 시 캐시되어 repeated polling에도 동일하다.

### Success — 수집 원문

```json
{
  "messageType": "FINAL_RESULT",
  "sessionId": "spec-60bd4137",
  "trigger": {
    "triggerId": "spec-trigger",
    "triggerType": "KEYPAD",
    "securityEventId": null
  },
  "analysisState": "COMPLETED",
  "analysisCompleteness": "COMPLETE_KEYPAD",
  "relatedEarlyEventInstanceIds": [],
  "status": "success",
  "result": {
    "visionEvents": [],
    "trackedPersonCount": 0,
    "streamingPocScope": "KEYPAD Silent+Rear or AUDIO Occlusion+Dwell when COMPLETE"
  },
  "error": null,
  "emittedAt": "2026-11-02T14:49:01.623+09:00"
}
```

### Partial — 수집 원문 (15 FPS, KEYPAD 175204)

```json
{
  "messageType": "FINAL_RESULT",
  "sessionId": "fps-15-7c6c03bd",
  "trigger": {
    "triggerId": "fps-test-15.0",
    "triggerType": "KEYPAD",
    "securityEventId": null
  },
  "analysisState": "COMPLETED",
  "analysisCompleteness": "PARTIAL_SAMPLED_TIME_AXIS",
  "relatedEarlyEventInstanceIds": ["fps-15-7c6c03bd-signal-001"],
  "status": "partial",
  "result": {
    "visionEvents": [
      {
        "eventType": "SILENT_SIGNAL",
        "detectedFrame": 114,
        "confirmedFrame": 114,
        "capturedAt": "3.799464"
      }
    ],
    "trackedPersonCount": null,
    "streamingPocScope": "KEYPAD Silent+Rear or AUDIO Occlusion+Dwell when COMPLETE"
  },
  "error": null,
  "emittedAt": "2026-11-02T14:38:52.097+09:00"
}
```

### Failed — 수집 원문

```json
{
  "messageType": "FINAL_RESULT",
  "sessionId": "failed-49051231",
  "trigger": {
    "triggerId": "failed-trigger",
    "triggerType": "INVALID",
    "securityEventId": null
  },
  "analysisState": "FAILED",
  "analysisCompleteness": "FAILED",
  "relatedEarlyEventInstanceIds": [],
  "status": "failed",
  "result": {
    "visionEvents": [],
    "trackedPersonCount": null,
    "streamingPocScope": "KEYPAD Silent+Rear or AUDIO Occlusion+Dwell when COMPLETE"
  },
  "error": "unsupported triggerType=INVALID",
  "emittedAt": "2026-11-02T14:49:06.052+09:00"
}
```

FINAL_RESULT 필드 타입:

| 위치 | 타입 | nullable |
|---|---|---:|
| `messageType` | string (`FINAL_RESULT`) | 아니오 |
| `sessionId` | string | 아니오 |
| `trigger.triggerId`, `trigger.triggerType` | string | 아니오 |
| `trigger.securityEventId` | string \| null | 예 |
| `analysisState`, `analysisCompleteness`, `status` | string | 아니오 |
| `relatedEarlyEventInstanceIds` | array<string> | 아니오 |
| `result` | object | 아니오 |
| `result.visionEvents` | array<object> | 아니오 |
| `result.trackedPersonCount` | integer \| null | 예 |
| `result.streamingPocScope` | string | 아니오 |
| `error` | string \| null | 예 |
| `emittedAt` | string | 아니오 |

## 7. 오류 응답 원문

모든 아래 오류 body는 FastAPI 기본 JSON object이며 `detail`은 string이다.

| HTTP | 발생 조건 | 수집/구현 원문 |
|---:|---|---|
| 404 | 알 수 없거나 TTL 만료 session | `{"detail":"unknown sessionId"}` |
| 409 | frame index gap | `{"detail":"out-of-order frame: expected 1"}` |
| 409 | complete 후 frame | `{"detail":"session input is already complete"}` |
| 409 | sessionId 중복 | `{"detail":"sessionId already exists"}` |
| 422 | triggerId/eventId 모두 없음 | `{"detail":"triggerId is required (eventId is accepted only as a legacy alias)"}` |
| 422 | AUDIO에서 securityEventId 없음 | `{"detail":"securityEventId is required for AUDIO"}` |
| 429 | worker queue 포화 | `{"detail":"session queue is full; retry this frame"}` |
| 415 | JPEG 이외 content type | `{"detail":"frame must be image/jpeg"}` |
| 400 | callback URL scheme/host 불허 | `{"detail":"callbackUrl host is not in STREAMING_POC_CALLBACK_HOSTS allowlist"}` |
| 202 | result worker drain 중 | `{"detail":"analysis is still draining"}` |

Pydantic 타입 검증 실패(예: `sourceFps` 누락, string 대신 object)는 FastAPI 표준 422의 `detail: [...]` 배열 형식이며, 위의 수동 422 string detail과 형태가 다를 수 있다.

## Spring Boot DTO 구현 주의

1. `analysisState`와 `analysisCompleteness`를 같은 enum으로 만들지 말 것.
2. `result.trackedPersonCount`는 primitive `int`가 아니라 nullable `Integer`여야 한다.
3. `securityEventId`, `error`, `capturedAt`, `completedSessionTtlSecondsRemaining`도 nullable이다.
4. `visionEvents`는 eventType별 details 구조가 달라 generic object/tree 또는 polymorphic DTO가 필요하다.
5. `status=partial`은 HTTP 성공(200)이지만 fallback이 필요한 분석 범위다.
6. `GET /result` HTTP 202은 FINAL_RESULT object가 아닌 `detail` error object다.
