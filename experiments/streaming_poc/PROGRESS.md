# OmniGuardy Vision Streaming PoC — Progress

## Scope locked for this run

Phase 1 only: frame-by-frame online YOLO/BoT-SORT tracking smoke test and
comparison with stored full64 MP4 tracking records. No existing Vision API,
tracking rules, model weights, video, or production file is modified.

## Planned Phase 1 checks

- `YOLO.track(frame, persist=True)` keeps tracking state within one session.
- An explicit session finalize resets the predictor's tracker before reuse.
- Online records are compared with stored full64 records for bbox IoU, track-ID
  correspondence, and per-track continuity.
- Representative KEYPAD cases: `175204` and `174955`.

## Phase 1 result — completed

| Case | Frames | Online FPS | Boxes / full64 boxes | Mean IoU | Same track-ID rate |
|---|---:|---:|---:|---:|---:|
| `175204` | 397 | 12.8030 | 477 / 477 | 1.00000 | 1.00000 |
| `174955` | 414 | 13.2735 | 743 / 743 | 1.00000 | 1.00000 |

- The PoC used the existing `tracking/yolov8n.pt`, `botsort.yaml`, `conf=0.3`,
  `classes=[0]`, and `imgsz=640`, with no model or tracker configuration edit.
- Each session invoked `model.track(frame, persist=True)` for every decoded
  source frame. The output exactly matched the stored full64 records for these
  two cases under an IoU >= 0.5 matching criterion.
- `finalize()` explicitly called `reset()` on the predictor's one BoT-SORT
  tracker. A following session restarted IDs at `1`, as expected. Track IDs
  are session-local and must therefore always be paired with `sessionId`.
- Details: `experiment_results/streaming_poc/phase1/` and
  `experiment_results/streaming_poc/phase1_174955/`.

## Explicitly deferred

- JPEG encode/decode and network simulation
- 30/15/10 FPS, resolution, queue/backlog, and real-time replay tests
- MP4 recording branch concurrency
- Online MediaPipe / final Silent Signal parity
- AUDIO rule parity
- Streaming FastAPI implementation and Spring Boot integration

## Phase 2 result — completed, limited matrix

The Phase 2 benchmark is in `experiment_results/streaming_poc/phase2/`.
It covers the requested staged set on `175204`, plus one candidate condition on
`174955`. It intentionally does not attempt the full FPS × resolution × JPEG
matrix.

- 30 FPS, 960x540 JPEG no-drop replay reached a maximum queue of 77 frames
  and required 2.7314 seconds after capture ended to drain.
- At 15 FPS and 10 FPS (960x540 JPEG), every submitted frame completed before
  capture ended, though the 15-FPS run still had a startup-induced maximum
  queue of 51 frames.
- 640x360 JPEG improved raw tracking throughput, but this phase did not test
  MediaPipe hand landmarks or AUDIO occlusion at that resolution. It is not a
  final camera specification.

## Phase 3 smoke result — partial

- `phase3_dual_path_smoke.py` confirmed one virtual 30-FPS source can feed a
  complete local MP4 writer and a JPEG YOLO worker concurrently.
- On `175204`, 30-FPS JPEG retained a Rear Approach event but left a 3.54-second
  post-capture Vision drain; 10 FPS finished at capture end but did not retain
  that Rear Approach event.
- `phase3_silent_signal_smoke.py` retained the existing 175204 candidate
  Signal frame at 30/15/10 FPS, without enabling either A/B experiment flag.
- AUDIO online parity is deliberately still unverified. Phase 3 did not modify
  path-based exhibition rules to make an untested claim of equivalence.

## Phase 4 limited parity result

- AUDIO original-resolution 30-FPS frame-state parity passed on representative
  occlusion (`173225`) and dwell (`173843`) cases; 960x540 with unchanged
  occlusion thresholds yielded no events.
- On 175204, Rear Approach survived 15 FPS but not 10 FPS. Silent Signal alone
  survived 30/15/10 on this candidate; joint YOLO+MediaPipe session parity is
  still a required follow-up.

## Safety

- Output: `experiment_results/streaming_poc/phase1/`
- No commit, push, reset, restore, clean, or deletion is performed.

## Early-event callback PoC result (completed)

- Independent source/code: `experiments/streaming_poc/early_event_poc/`; results:
  `experiment_results/streaming_poc/early_event/`.
- `175204` replay confirmed the unchanged legacy Silent Signal state machine at
  source frame 114. `detectedFrame` and `confirmedFrame` are both 114 because
  the event is emitted at the frame where the tracker condition returns true;
  it is not a post-video backfill.
- A localhost mock callback received the provisional event about 4.21 seconds
  after capture start, before 13.23-second capture completion and before YOLO
  finished at 16.66 seconds. A deliberate HTTP 500 was retried once using the
  same event ID and sequence, then accepted with HTTP 204.
- The unchanged default rule also yielded known human-reviewed-negative false
  candidates (`174439`: 35 and 278; `174454`: 174). Early delivery therefore
  proves timing/transport only; it does not make the current detector safe for
  unattended alerting. Details and API-contract decisions are in
  `experiment_results/streaming_poc/early_event/EARLY_EVENT_REPORT.md`.
