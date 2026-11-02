# Pi ↔ Streaming Vision ↔ Spring Boot readiness contract

This is the standalone streaming service at port **8010**. It does not replace
the protected MP4 API at port 8000. The validated representative input is a
7-second capture: KEYPAD `960x540`, JPEG quality 85, 30 FPS; AUDIO
`1920x1080`, JPEG quality 85, 30 FPS.

## Raspberry Pi sequence

1. Create one UUID `sessionId` for one camera trigger. Pi sends the same value
   to Vision and registers it with Spring Boot before frame transmission.
2. Call `POST http://VISION_HOST:8010/vision/start`.
3. Start one camera capture. In separate worker paths, write the 30-FPS MP4 for
   Spring Boot/S3 and JPEG-encode/send frames to Vision. Neither path may wait
   for the other.
4. Send exactly contiguous transmit `frameIndex` values to `/vision/frame`.
5. After the camera input closes, call `/vision/complete`; it means input closed
   only, not final analysis complete.
6. Spring Boot polls status then result. It decides whether to call Agent once
   or invoke the existing MP4 fallback once.

`sessionId` is Pi-created. `triggerId` is the existing trigger identity and
must not be recreated by Vision. `securityEventId` is required for AUDIO.
`eventId` remains a legacy alias only.

```json
POST /vision/start
{
  "sessionId": "c3a5fcea-0d3b-4b0a-ae31-617e2cf78001",
  "triggerId": "existing-trigger-id",
  "securityEventId": "required-for-audio-only",
  "triggerType": "KEYPAD",
  "sourceFps": 30.0,
  "callbackUrl": "https://spring.internal/api/vision/early-event",
  "finalResultCallbackEnabled": false
}
```

```text
POST /vision/frame   Content-Type: multipart/form-data
sessionId=<same UUID>
frameIndex=57              # transmission order: 0, 1, 2 ... no gap
sourceFrameIndex=57        # original camera frame; same as above at 30 FPS
capturedAt=2026-11-02T14:00:01.900+09:00
frame=@frame.jpg;type=image/jpeg
```

`sourceFrameIndex` and `capturedAt` describe the original camera time axis;
they are not substitutes for skipped observations. The present final-analysis
contract requires continuous 30-FPS input. Pi must not silently downsample to
15/20 FPS and expect a complete final result.

On `429`, retry the **same** frame index and JPEG with bounded backoff; do not
advance the index. On `409`, query status and correct the sender state: it
indicates an index gap/out-of-order request (or input already complete). A
duplicate earlier index receives `202` with `status:"duplicate"` and must not
be generated again. If network transmission cannot recover during capture,
finish MP4 recording and mark streaming unavailable for fallback.

```json
POST /vision/complete
{"sessionId":"c3a5fcea-0d3b-4b0a-ae31-617e2cf78001"}
```

Pi performance is **not** validated: the local replay proves the Vision API
contract, not that a Pi can JPEG encode and directly deliver all 210 frames in
7 seconds on the deployment network.

## Spring Boot callback and polling

Vision only accepts callback hosts listed in server-side
`STREAMING_POC_CALLBACK_HOSTS`; configure the known internal Spring hostname.
Do not allow an arbitrary Pi-supplied URL. The server can append an internal
Bearer value from `STREAMING_POC_CALLBACK_BEARER_TOKEN`; production must agree
on mTLS or a signed/internal-service credential before deployment.

```json
{
  "messageType":"EARLY_EVENT",
  "sessionId":"c3a5fcea-0d3b-4b0a-ae31-617e2cf78001",
  "triggerId":"existing-trigger-id",
  "securityEventId":null,
  "triggerType":"KEYPAD",
  "eventInstanceId":"c3a5fcea-0d3b-4b0a-ae31-617e2cf78001-signal-001",
  "sequence":1,
  "provisional":true,
  "event":{"eventType":"SILENT_SIGNAL","detectedFrame":114,"confirmedFrame":114,"capturedAt":"..."},
  "emittedAt":"..."
}
```

Persist/deduplicate callback deliveries by `(sessionId, eventInstanceId)`.
Retry retains exactly the same eventInstanceId and sequence; a distinct signal
receives a new instance ID. EARLY_EVENT is a recorded provisional observation;
under the agreed policy it must not execute Agent or user push by itself.

Poll `GET /vision/status/{sessionId}` every 250–500 ms initially, then apply
bounded backoff. Obtain `GET /vision/result/{sessionId}` only after terminal
status. `/vision/complete` returning 202 is not terminal.

```json
{
  "messageType":"FINAL_RESULT",
  "sessionId":"...",
  "trigger":{"triggerId":"existing-trigger-id","triggerType":"KEYPAD","securityEventId":null},
  "analysisState":"COMPLETED",
  "analysisCompleteness":"COMPLETE_KEYPAD",
  "relatedEarlyEventInstanceIds":["...-signal-001"],
  "status":"success",
  "result":{"visionEvents":["existing event objects"],"trackedPersonCount":3}
}
```

Interpret `analysisState` only as lifecycle: `RECEIVING`, `PROCESSING`,
`COMPLETED`, `FAILED`. Interpret `analysisCompleteness` as coverage:
`COMPLETE_KEYPAD`, `COMPLETE_AUDIO`, `PARTIAL_SAMPLED_TIME_AXIS`, `FAILED`.
Only `COMPLETED` + `status=success` + matching COMPLETE value can take the
streaming final route. `partial`, `FAILED`, timeout, or unrecoverable upload
error requires exactly one MP4 fallback for the same `triggerId`; Agent must
then run only on that selected final route.

## Service settings still needed before real integration

- Bind 8010 on the actual Vision host, while port 8000 remains the MP4 fallback.
- Set the callback host allowlist and authentication secret/certificate.
- Set queue capacity and completed-result TTL deliberately. Defaults are 300
  frames per worker queue and 3600 seconds. `STREAMING_POC_QUEUE_CAPACITY` and
  `STREAMING_POC_COMPLETED_SESSION_TTL_SECONDS` are server-side settings.
- Completed sessions are removed only when a later API request invokes cleanup;
  active sessions are deliberately never auto-deleted. A production deployment
  still needs a background cleanup/input-idle timeout policy and persistent
  session/result storage.
- Add observability for queue depth, callback failures, GPU memory, timeouts,
  and fallback decisions.
- Run a real Pi 210-frame / 7-second delivery test and Spring Boot authentication
  test before enabling streaming as the Agent-final path.

## Coverage boundary

At validated contiguous 30 FPS, this PoC has baseline Silent Signal + Rear
Approach (KEYPAD), and Occlusion + Dwell (AUDIO). Optional legacy
`WEAPON_PROXIMITY_SUSPECTED` is not in the streaming adapter. If
`VISION_ENABLE_WEAPON_DETECTION=true` is used in the MP4 deployment, streaming
must not be labelled a complete KEYPAD replacement; select MP4 fallback or add
and separately validate a stream weapon adapter. O1 ownership remains excluded.
