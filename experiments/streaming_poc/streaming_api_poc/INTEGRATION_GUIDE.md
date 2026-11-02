# Raspberry Pi ↔ Streaming Vision API contract (PoC)

This is a standalone service. Do not replace the MP4 API at port 8000.
Suggested local test address: `http://VISION_HOST:8010`.

## Identity contract

- Raspberry Pi creates `sessionId` once per camera trigger (UUID recommended) and sends it to Vision first, then registers that exact value with Spring Boot.
- `triggerId` is the existing Vision trigger identifier. Do not create a second event identifier in Vision.
- `securityEventId` is passed through for AUDIO and is required by this PoC when `triggerType=AUDIO`.
- `eventId` is only a backward-compatible PoC alias. New callers use `triggerId`.

If Pi cannot start/register a session before capture, it still records the MP4. Spring Boot marks streaming unavailable and invokes existing MP4 analysis once after recording/upload completes.

## Pi calls

Start:

```json
POST /vision/start
{
  "sessionId": "pi-uuid",
  "triggerId": "existing-trigger-id",
  "triggerType": "KEYPAD",
  "sourceFps": 30.0,
  "callbackUrl": "https://spring.internal/vision/early-event",
  "finalResultCallbackEnabled": false
}
```

`callbackUrl` is accepted only if its host is in the Vision deployment's
`STREAMING_POC_CALLBACK_HOSTS` allowlist. The service may add an Authorization
Bearer token from `STREAMING_POC_CALLBACK_BEARER_TOKEN`; Pi never provides it.

Frame, sent as multipart:

```text
POST /vision/frame
sessionId=pi-uuid
frameIndex=57              # contiguous transmit sequence: 0, 1, 2...
sourceFrameIndex=114       # original 30-FPS camera frame number
capturedAt=2026-11-02T14:00:00.000+09:00
frame=@frame.jpg;type=image/jpeg
```

For 20 FPS analysis from a 30 FPS camera, `frameIndex` remains contiguous but
`sourceFrameIndex` advances according to the captured source frame. MP4
recording and JPEG transmission run on separate producer/worker paths; neither
waits for the other.

On `429`, retry that exact frame with bounded backoff. On `409`, query status;
do not skip ahead or reuse a new frame index. Network loss is a streaming
failure candidate, not a reason to silently drop frames.

Finish:

```json
POST /vision/complete
{"sessionId":"pi-uuid"}
```

`202` means only that input is closed.

## Spring Boot calls and callbacks

Status response includes the lifecycle `analysisState`: `RECEIVING`,
`PROCESSING`, `COMPLETED`, `FAILED`; `finalResultAvailable`; `triggerId`;
`sessionId`; and an error on failure. `analysisCompleteness` is deliberately a
separate field: `COMPLETE_KEYPAD`, `COMPLETE_AUDIO`,
`PARTIAL_SAMPLED_TIME_AXIS`, or `FAILED`. Poll
`GET /vision/status/{sessionId}`, then `GET /vision/result/{sessionId}` only
after lifecycle state `COMPLETED`.

EARLY_EVENT example:

```json
{
  "messageType":"EARLY_EVENT",
  "sessionId":"pi-uuid",
  "triggerId":"existing-trigger-id",
  "securityEventId":null,
  "triggerType":"KEYPAD",
  "eventInstanceId":"pi-uuid-signal-001",
  "sequence":1,
  "provisional":true,
  "event":{"eventType":"SILENT_SIGNAL","detectedFrame":114,"confirmedFrame":114,"capturedAt":"..."},
  "emittedAt":"..."
}
```

Deduplicate callback retries by `(sessionId, eventInstanceId)`. The same
instance/sequence is retried; a genuinely independent signal receives a new
instance ID and sequence. Record EARLY_EVENT separately. Do not invoke Agent
AI from it under the current policy.

FINAL_RESULT is obtained by polling. Its `relatedEarlyEventInstanceIds` links
to the early records. Spring Boot invokes Agent AI exactly once for a completed
`triggerId`; on streaming failure it invokes the MP4 fallback exactly once for
that same trigger, never both final paths.

The final payload repeats lifecycle `analysisState` and provides
`analysisCompleteness` separately. `status=success` is eligible for the
streaming final path only when completeness is `COMPLETE_KEYPAD` or
`COMPLETE_AUDIO`. `status=partial` / `PARTIAL_SAMPLED_TIME_AXIS` means frames
were intentionally sampled: only a directly observed online Silent Signal may
be present, while Rear Approach, Camera Occlusion, and Person Dwell are not
claimed. Spring Boot must choose the MP4 fallback for that trigger unless a
future, separately validated sampled-time-axis adapter is deployed.

For contiguous 30-FPS source frames the PoC supports baseline Silent Signal +
Rear Approach for KEYPAD, and Occlusion + Dwell for AUDIO. It remains a PoC:
weapon events, sampled-time-axis parity, full-corpus parity, production
session storage, and actual Pi/Spring Boot transport are not complete.
