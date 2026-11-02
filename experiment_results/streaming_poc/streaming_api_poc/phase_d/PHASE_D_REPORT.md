# Streaming API Phase D — local integration readiness verification

## Boundaries

All tests used localhost mock HTTP only. No Raspberry Pi, Spring Boot,
authentication credential, S3, MQTT, or port-8000 MP4 API was contacted. The
protected MP4 API, detectors, tracking rules, final-demo code, videos, and
model weights were not changed.

## Implemented streaming-only hardening

- FINAL_RESULT payload is cached after terminal completion, so repeated
  `/vision/result/{sessionId}` reads are byte-for-byte stable for the same
  session instead of regenerating `emittedAt`.
- Queue capacity is server-configurable with
  `STREAMING_POC_QUEUE_CAPACITY` (default 300 per worker queue).
- `/vision/complete` closes input and asynchronously enqueues worker sentinels.
  It returns 202 even when an accepted-frame queue is full; accepted frames are
  not discarded before the sentinel.
- Completed session result retention is configurable with
  `STREAMING_POC_COMPLETED_SESSION_TTL_SECONDS` (default 3600 seconds). On a
  later API request, expired terminal sessions are released and return 404.
  RECEIVING/PROCESSING sessions are intentionally retained; production idle
  timeout/periodic cleanup is still missing.

## One end-to-end mock flow — KEYPAD 175204

The Pi mock sent 210 contiguous frames over 6.994 seconds (960x540, JPEG Q85,
30 FPS) directly to port 8010. A local callback receiver intentionally returned
HTTP 500 once, then 204.

| Check | Measured result |
|---|---|
| FINAL_RESULT lifecycle / coverage | `COMPLETED` / `COMPLETE_KEYPAD` |
| Final status | `success` |
| Events | Silent Signal 114; Rear Approach 43; tracked persons 3 |
| POST-complete drain | 1.364 s |
| Capture start to final result | 8.357 s |
| Max queues (MediaPipe / YOLO) | 16 / 34 |
| Callback attempts | 2: first 500, retry 204 |
| Retry identity | same `eventInstanceId` and `sequence` |
| Early/Final relation | Silent event in FINAL_RESULT and ID in `relatedEarlyEventInstanceIds` |
| Result polling idempotence | two GETs returned identical FINAL_RESULT payloads |

The raw record is [integration_contract_smoke.json](integration_contract_smoke.json).
This is a single localhost run; it is not a Pi network throughput measurement.

## Contract/error tests

| Case | Actual result | Contract implication |
|---|---:|---|
| unknown status session | 404 | Caller treats as unknown/expired session. |
| missing `triggerId` | 422 | Start validation fails before session creation. |
| duplicate `sessionId` | 409 | Do not create another stream for the same trigger. |
| first frame index 1 | 409 | Pi must begin with contiguous `frameIndex=0`. |
| retransmit accepted frame 0 | 202, `duplicate` | Safe idempotent retransmission. |
| duplicate `/complete` | 202 both times | Input closure is idempotent. |
| frame after complete | 409 | Pi must stop frame uploads. |
| callback receiver 500 | retry then 204 | Retry keeps the same event identity. |
| unsupported trigger worker finalization | `FAILED` / `status=failed` | Spring Boot can select fallback from terminal result. |
| queue capacity 2 probe | 202, 202, 429, then 409 when sender incorrectly advanced index | Retry index 2, never skip to index 3. |
| full-queue `/complete` | 202 in 12.973 ms | Closure does not wait for worker queue drain. |
| terminal TTL 1s probe | status became 404 after expiry-triggering status request | Terminal resource cleanup works on subsequent API traffic. |

The main flow was preceded by a separate completed probe session, with unique
session/trigger IDs and empty tracks; no response/event state crossed into the
main session. This demonstrates sequential session isolation only. Concurrent
GPU multi-session throughput/isolation is not yet measured.

## Fallback decision

Spring Boot records EARLY_EVENT separately. It invokes Agent exactly once only
for `analysisState=COMPLETED`, `status=success`, and the trigger-appropriate
`analysisCompleteness` (`COMPLETE_KEYPAD` / `COMPLETE_AUDIO`). `FAILED`,
`PARTIAL_SAMPLED_TIME_AXIS`, expired session, polling timeout, or failed uploads
select exactly one MP4 fallback for that same triggerId. Never execute Agent on
both streaming and fallback result paths.

## Outstanding operational risks

- Pi 30-FPS JPEG delivery for 210 frames/7 seconds remains unmeasured.
- Production callback authentication, TLS/mTLS, reverse proxy limits, and
  Spring Boot polling/backoff are not validated.
- No persistent session/result store, background cleaner, receiver acknowledgement
  ledger, or abandoned-session input timeout exists.
- 30-FPS event parity is representative, not corpus-wide; 15/20 FPS remains
  partial by design.
- Streaming has no optional legacy Weapon proximity adapter. O1 ownership is
  intentionally not integrated.
