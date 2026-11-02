# Streaming Vision API Phase B — implemented scope and verification

## Implementation

The standalone streaming server now starts three session-local workers as applicable:

- MediaPipe worker: baseline Silent Signal and immediate EARLY_EVENT.
- YOLOv8n + BoT-SORT worker: `model.track(frame, persist=True)`, one YOLO model/tracker per session, rows appended as `frame, track_id, x1, y1, x2, y2, confidence`.
- AUDIO worker: online Canny/Laplacian occlusion metrics.

Workers receive the same incoming JPEG bytes through separate queues. MediaPipe does not wait for YOLO drain. The finalizer runs only after every relevant worker completes.

For contiguous source-frame input only, it calls the existing rules without changing them:

- KEYPAD: `tracking.keypad_rules_front.detect_front_rear_approach()` over the collected tracking DataFrame.
- AUDIO: frame adapter with the existing occlusion thresholds, and `exhibition_audio_service.detect_person_dwell()` over the collected tracking DataFrame.

For a sampled source time axis, final status is `partial` / `PARTIAL_SAMPLED_TIME_AXIS`; the API does not invent Rear/Dwell/Occlusion output from incomplete temporal input.

## FINAL_RESULT Silent Signal inclusion

Verified: an EARLY_EVENT emits one session event instance, and the same event is present once in `FINAL_RESULT.result.visionEvents`. `relatedEarlyEventInstanceIds` contains the corresponding ID. The finalizer appends Rear/AUDIO events separately and does not create a second Silent event.

`FINAL_RESULT.status=success` only with `analysisState=COMPLETE_KEYPAD` or `COMPLETE_AUDIO`. Worker failure returns `status=failed` and error detail. A sampled timeline returns `status=partial`.

## HTTP verification

All replay clients used direct local HTTP to the separate server at port 8012; the process was stopped after tests. No port-8000 MP4 API was modified or used.

| Case | Input | Streaming final events | Existing reference | Comparison |
|---|---|---|---|---|
| KEYPAD `175204`, first 7s | 960x540 JPEG, 30 FPS, 210 contiguous source frames | Silent 114; Rear 43; persons 3 | full64: Silent 114; Rear 44; max people 3 | Silent/count match; Rear differs 1 frame |
| KEYPAD `175021`, full 12.96s | 960x540 JPEG, 30 FPS, 389 contiguous frames | Silent 335; Rear 284; persons 4 | full64: Silent 333; Rear 284; max people 4 | Rear/count match; Silent differs 2 frames |
| AUDIO occlusion `173225`, first 7s | 1920x1080 JPEG, 30 FPS, 210 contiguous frames | Occlusion 162, 174, 207 | previous JPEG parity: 162, 174, 207 | match |
| AUDIO dwell `173843`, first 7s | 1920x1080 JPEG, 30 FPS, 210 contiguous frames | Dwell 150, track 1 | previous JPEG parity: Dwell 150, details track 1 | detected frame/track match |

The Phase B API intentionally still uses the baseline Silent Signal rule. O1 ownership is not integrated because its raw `175021` miss and `174439` residual FP remain unresolved.

## 7-second timing, 30FPS after YOLO addition

| Case | MediaPipe max queue | YOLO max queue | Audio max queue | Drain after complete | Capture→final |
|---|---:|---:|---:|---:|---:|
| KEYPAD `175204` | 16 | 61 | n/a | 2.241 s | 9.258 s |
| AUDIO `173225` | 73 | 131 | 48 | 10.551 s | 17.744 s |
| AUDIO `173843` | 85 | 128 | 36 | 8.975 s | 15.984 s |

No GPU memory or CPU utilization sample was captured. 15/20 FPS was tested before YOLO integration only; it has **not** been revalidated with this Phase B worker and must not be treated as a final setting.

## Result-file note

The first Phase B KEYPAD replay mistakenly reused the older root
`fps_30_7s.json` filename before the client output path was corrected. The
verified Phase B result is stored separately as `keypad_30_7s.json` in this
directory. The old root file had no automatic backup in this workspace.

## Remaining limits

- No 15/20 FPS Phase B parity for Rear/AUDIO yet.
- No Session TTL/cleanup store, production metrics, concurrency limit, or persistent session storage.
- Existing weapon event is not implemented in the adapter.
- Audio rule parity is established only for two representative 30FPS JPEG cases, not the complete corpus.
- Callback/HTTP loopback is tested; actual Pi and Spring Boot transport/authentication are not.
