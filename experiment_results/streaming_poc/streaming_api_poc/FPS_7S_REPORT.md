# 7-second Streaming API FPS replay — Phase A/C partial result

## What was implemented and tested

`experiments/streaming_poc/streaming_api_poc/app.py` is an independent FastAPI server. It does not import or modify `api.vision_api.py`, and was tested only on localhost port 8011 (then stopped).

Implemented:

- `POST /vision/start`, `/vision/frame`, `/vision/complete`
- `GET /vision/status/{sessionId}`, `/vision/result/{sessionId}`
- Pi-supplied `sessionId`; canonical `triggerId` plus optional `securityEventId`; old PoC `eventId` is accepted as a deprecated alias.
- Separate transmit `frameIndex` and original-camera `sourceFrameIndex`. MediaPipe timestamps use `sourceFrameIndex / sourceFps`, so sampled JPEG input retains its original capture time axis.
- Causal baseline Silent Signal online tracker, asynchronous EARLY_EVENT callback, idempotent event instance IDs, optional FINAL_RESULT callback (off by default), and polling final result.
- State response: `RECEIVING`, `PROCESSING`, `COMPLETED`, or `FAILED`, plus final-result availability, callback records and queue maximum.

Not implemented in this server: online YOLO/BoT-SORT, Rear Approach, AUDIO occlusion/dwell, tracked person count, or full legacy `visionEvents` parity. `trackedPersonCount` is explicitly `null`. O1 is not integrated.

## Replay conditions

- Source: `175204`, first 7.0 seconds only (source frames 0–209/210)
- JPEG: 960x540, quality 85
- Source camera clock: 30.004 FPS
- Client: real-time HTTP replay, not maximum-throughput submission
- One run per setting, localhost callback 204, no GPU/YOLO worker in this API
- The Signal around source frame 114 (about 3.80 s) is expected in this 7-second window.

## Measured result

| JPEG transmit FPS | Accepted / sent frames | 409 / 429 / duplicate | Signal confirmed source frame | Capture→callback | Max MediaPipe queue | Drain after complete | Capture→final |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 15 | 105 / 105 | 0 / 0 / 0 | 115 (3.833 s) | 3.904 s | 1 | 0.233 s | 7.236 s |
| 20 | 140 / 140 | 0 / 0 / 0 | 115 (3.833 s) | 3.891 s | 2 | 0.123 s | 7.134 s |
| 30 | 209 / 209 | 0 / 0 / 0 | 114 (3.799 s) | 4.201 s | 19 | 0.235 s | 7.217 s |

The 15/20 FPS event moved one source frame because sampled input omitted source frame 114 and included 115. It did not prove full gesture equivalence on a labelled dataset. The 30 FPS callback was later despite the earlier source event because the MediaPipe queue accumulated.

## Recommendation boundary

For this Silent-Signal-only 7-second localhost test, **20 FPS / 960x540 / quality 85** is the balanced candidate: all frames were accepted, queue depth stayed 2, final result was available 0.123 seconds after completion, and the signal was confirmed at 3.833 seconds. This is not a final KEYPAD FPS recommendation because Rear Approach was not executed. It is not an AUDIO recommendation: AUDIO must remain 1920x1080 and occlusion/dwell parity at 15/20/30 FPS is unmeasured.

No YOLO queue, GPU-memory, CPU utilization, Rear Approach, Camera Occlusion, Dwell, or MP4 parity metric exists for this run. They must be reported as unmeasured, not zero.

## Error/fallback policy proposal

- Pi must retry the same frame on `429` with bounded backoff; do not advance `frameIndex`.
- Pi must not silently discard a `409`: query status, compare `nextExpectedFrameIndex`, retry only if the server has not already accepted the index.
- `/vision/complete` returning `202` means input closed only. Spring Boot polls status/result until `COMPLETED` or `FAILED`.
- Suggested polling: 250–500 ms initially, then 1 s; bounded 30 s after a 7-second capture. On `FAILED`, timeout, lost session, or incomplete upload, mark the one `triggerId` as streaming-failed and use the saved MP4 for the existing `/analyze/vision` fallback exactly once.
- Spring Boot must deduplicate final processing/fallback by `triggerId` (and `securityEventId` for AUDIO), so Agent AI is invoked once.
