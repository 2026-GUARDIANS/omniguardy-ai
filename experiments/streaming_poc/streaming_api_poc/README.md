# OmniGuardy Streaming Vision API PoC

이 서버는 기존 MP4 Vision API(`api.vision_api`, 포트 8000)를 변경하지 않고
별도 포트 8010에서 실행하는 Streaming PoC입니다.

```powershell
.\.venv\Scripts\python.exe -m uvicorn experiments.streaming_poc.streaming_api_poc.app:app --host 0.0.0.0 --port 8010
```

## 역할과 검증 범위

Raspberry Pi는 한 번의 7초 카메라 촬영에서 JPEG를 Vision에 직접 전송하고, MP4
녹화는 Spring Boot/S3 보관 경로로 병렬 처리합니다. Vision은 KEYPAD의 baseline
Silent Signal(조기 callback)·Rear Approach, AUDIO의 Camera Occlusion·Person Dwell을
세션별 worker에서 처리합니다. 최종 결과는 callback이 아니라 Spring Boot polling
으로 조회합니다.

| API | 역할 |
|---|---|
| `POST /vision/start` | Pi가 만든 `sessionId`와 기존 `triggerId`로 세션 생성 |
| `POST /vision/frame` | ordered multipart JPEG 전송 (`frameIndex`, `sourceFrameIndex`, `capturedAt`) |
| `POST /vision/complete` | 입력 종료만 표시; HTTP 202은 분석 완료가 아님 |
| `GET /vision/status/{sessionId}` | `RECEIVING`/`PROCESSING`/`COMPLETED`/`FAILED` 확인 |
| `GET /vision/result/{sessionId}` | terminal FINAL_RESULT 조회 |

`EARLY_EVENT`는 Silent Signal 확인 시 비동기 callback으로 전송하며 retry에도 같은
`eventInstanceId`와 `sequence`를 사용합니다. FINAL_RESULT는 `analysisState`와
`analysisCompleteness`를 구분합니다. `partial`, `FAILED`, timeout, 업로드 실패는
같은 `triggerId`의 기존 MP4 `/analyze/vision` fallback 대상입니다.

## 현재 권장 입력과 제한

- KEYPAD: `960x540`, JPEG quality 85, 30 FPS
- AUDIO: `1920x1080`, JPEG quality 85, 30 FPS
- 15/20 FPS는 시간축이 샘플링되어 `PARTIAL_SAMPLED_TIME_AXIS`이며 최종 분석 대체용이 아님
- Optional Weapon proximity, O1 ownership 실험, 실제 Pi/Spring Boot 인증·네트워크는 미완료

7초 localhost 대표 검증에서 KEYPAD `175204`는 Silent Signal 114, Rear 43,
인원 3을 반환했고 capture 시작부터 FINAL_RESULT까지 약 8.36초였습니다. 이는
localhost 결과이며 Pi의 실제 210프레임/7초 전송 성능을 보장하지 않습니다.

## 환경변수

| 변수 | 기본값 | 용도 |
|---|---|---|
| `STREAMING_POC_CALLBACK_HOSTS` | `127.0.0.1,localhost` | callback URL host allowlist |
| `STREAMING_POC_CALLBACK_BEARER_TOKEN` | unset | Vision 서버가 callback에 넣는 Bearer token |
| `STREAMING_POC_QUEUE_CAPACITY` | `300` | worker별 frame queue 제한 |
| `STREAMING_POC_COMPLETED_SESSION_TTL_SECONDS` | `3600` | 완료 결과 보관 시간 |

실제 요청/응답·DTO 타입은
[Spring Boot JSON 명세](../../../experiment_results/streaming_poc/streaming_api_poc/SPRING_BOOT_JSON_SPEC.md),
Pi/Spring Boot 순서는 [연동 가이드](PI_SPRING_BOOT_READINESS.md)를 따른다.
