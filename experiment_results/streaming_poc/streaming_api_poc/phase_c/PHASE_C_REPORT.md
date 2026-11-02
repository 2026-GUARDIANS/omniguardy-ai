# Streaming API Phase C — sampled-FPS safety verification

## Scope and code protection

This verification used only the standalone service under
`experiments/streaming_poc/streaming_api_poc/` on localhost port 8013. It did
not start, call, or modify the MP4 Vision API on port 8000. No `api/`,
`tracking/`, detector, model-weight, final-demo-video, reset, restore, clean,
commit, or push operation was used.

The replay client was updated to retain two time axes and to avoid overwriting
prior reports:

- `frameIndex`: contiguous **transmit** order expected by `/vision/frame`.
- `sourceFrameIndex`: original camera-frame number used by MediaPipe timestamps,
  YOLO rows, `detectedFrame`, and rule time conversion.
- `capturedAt`: original capture timestamp carried through an early event.
- Results now use a per-video filename in this Phase C directory.

## Why 15/20 FPS is partial today

At 15/20 FPS from a 30 FPS source, `sourceFrameIndex` has gaps. The service
correctly identifies that with `source_is_contiguous()` and returns
`analysisCompleteness: PARTIAL_SAMPLED_TIME_AXIS`, `status: partial`.

This is intentional safety behavior, not a failed inference:

| Rule | Existing temporal assumption | Safe current sampled behavior |
|---|---|---|
| Silent Signal | Online MediaPipe state; source-frame timestamps supplied | Emit only an actually observed baseline Silent Signal. |
| Rear Approach | Per-frame YOLO tracks, entry order, proximity span, and distance closing | Do not run final rule on missing observations. |
| Person Dwell | Existing tracking rows and frame/FPS continuity | Do not run final rule on missing observations. |
| Camera Occlusion | Consecutive low-edge/low-Laplacian frames for a sustained interval | Do not claim a sustained sequence through unobserved frames. |

`capturedAt` and source indices preserve the temporal positions of observations,
but cannot reconstruct person/hand/occlusion evidence in skipped frames.
Converting a frame count to seconds alone would therefore not preserve legacy
rule semantics. No synthetic rows, interpolation, or inferred events are
created.

## Actual 7-second HTTP replay measurements

All inputs were JPEG quality 85 and replayed at the requested real-time
interval. The source segment was 210 original frames (about 7 seconds). Values
are single runs; GPU-memory/CPU-utilization samples were not captured, so they
are not reported as measurements.

| Trigger / video | Input | Sent / accepted | 409 / 429 | Media / YOLO / Audio max queue | Drain after complete | Final elapsed | FINAL_RESULT |
|---|---:|---:|---:|---:|---:|---:|---|
| KEYPAD `175204` | 30 FPS, 960x540 | 210 / 210 | 0 / 0 | 16 / 61 / n/a | 2.241 s | 9.258 s | `success`, `COMPLETE_KEYPAD`; Silent 114, Rear 43, persons 3 |
| KEYPAD `175204` | 20 FPS, 960x540 | 140 / 140 | 0 / 0 | 1 / 8 / n/a | 0.174 s | 7.182 s | `partial`; observed Silent 114 only |
| KEYPAD `175204` | 15 FPS, 960x540 | 105 / 105 | 0 / 0 | 2 / 18 / n/a | 0.125 s | 7.102 s | `partial`; observed Silent 114 only |
| AUDIO occlusion `173225` | 30 FPS, 1920x1080 | 210 / 210 | 0 / 0 | 73 / 131 / 48 | 10.551 s | 17.744 s | `success`, `COMPLETE_AUDIO`; occlusion 162/174/207 |
| AUDIO occlusion `173225` | 20 FPS, 1920x1080 | 140 / 140 | 0 / 0 | 4 / 59 / 2 | 5.121 s | 12.106 s | `partial`; no claim about occlusion |
| AUDIO occlusion `173225` | 15 FPS, 1920x1080 | 105 / 105 | 0 / 0 | 1 / 28 / 1 | 2.489 s | 9.495 s | `partial`; no claim about occlusion |
| AUDIO dwell `173843` | 30 FPS, 1920x1080 | 210 / 210 | 0 / 0 | 85 / 128 / 36 | 8.975 s | 15.984 s | `success`, `COMPLETE_AUDIO`; dwell 150, track 1 |
| AUDIO dwell `173843` | 20 FPS, 1920x1080 | 140 / 140 | 0 / 0 | 11 / 58 / 2 | 3.922 s | 10.911 s | `partial`; no claim about dwell |
| AUDIO dwell `173843` | 15 FPS, 1920x1080 | 105 / 105 | 0 / 0 | 1 / 18 / 1 | 1.186 s | 8.172 s | `partial`; no claim about dwell |

The 30-FPS rows are the existing Phase B local HTTP baselines. The new 15/20
FPS raw payloads are stored alongside this report. They all had zero duplicate,
out-of-order (`409`), or queue-full (`429`) responses.

For KEYPAD `175204`, early callback remained available at both sampled rates:
source `confirmedFrame=114`, `capturedAt=3.799464`; start-to-mock-receiver was
3.869 s at 20 FPS and 3.875 s at 15 FPS (30-FPS prior baseline: 4.154 s). Those
are localhost callback measurements, not Pi/Spring Boot latency. The 30/20/15
Silent result shows this one action was observed under sampling; it does **not**
prove sampled Silent-Signal parity for all gestures.

## Event parity and recommendation

The 30-FPS results retain the Phase B representative parity:

- `175204`: Silent Signal source frame 114 matches; Rear source frame 43 versus
  full64 reference frame 44; tracked-person count 3 matches.
- `173225`: JPEG-online occlusion events 162, 174, 207 match prior JPEG parity.
- `173843`: JPEG-online dwell source frame 150 / track 1 matches prior JPEG
  parity.

At 15/20 FPS, no Rear/Dwell/Occlusion parity is asserted because those event
families are intentionally not evaluated. Thus lower drain time is an input
reduction effect and is **not** a validated replacement for final Vision
analysis.

**Current safe configuration:** 30 FPS for KEYPAD at 960x540 Q85 and 30 FPS for
AUDIO at 1920x1080 Q85, subject to the representative-only parity scope above.
Do not select 15 or 20 FPS as the FINAL_RESULT source yet. They can only be
used as a separate experimental Silent-Signal early-observation path while the
MP4 fallback remains responsible for final coverage.

## Completion / fallback contract for Spring Boot

Use two independent fields:

- `analysisState`: lifecycle only — `RECEIVING`, `PROCESSING`, `COMPLETED`,
  `FAILED`.
- `analysisCompleteness`: analysis coverage — `COMPLETE_KEYPAD`,
  `COMPLETE_AUDIO`, `PARTIAL_SAMPLED_TIME_AXIS`, `FAILED`.

`/vision/complete` returns `202` only for input closure. Poll status until the
lifecycle is terminal. Execute Agent exactly once for a `triggerId` only when
`analysisState=COMPLETED`, `status=success`, and completeness is appropriate
for the requested trigger. If state is `FAILED`, or result is `partial` /
`PARTIAL_SAMPLED_TIME_AXIS`, record the streaming attempt and initiate exactly
one existing-MP4 fallback for the same triggerId. Deduplicate EARLY_EVENT
callback retries by `(sessionId, eventInstanceId)`; do not call Agent from an
early event under the current policy.

## Unimplemented coverage

`WEAPON_PROXIMITY_SUSPECTED` is an optional legacy MP4-only feature. In
`tracking/keypad_rules.py` it is gated by `VISION_ENABLE_WEAPON_DETECTION`
defaulting to false and runs a separate path-based YOLO `predict` pass for
COCO classes 43/76 before proximity calculation. The streaming adapter does
not implement that second detector/association pass. Consequently it is not a
complete KEYPAD replacement if that flag is enabled. O1 ownership and
same-hand experiments are also intentionally not enabled.

## Next validation required before sampled final analysis

1. Design a rule-by-rule sampled-time adapter that consumes actual timestamps,
   explicitly models observation gaps, and is tested against contiguous MP4
   references; do not simply scale thresholds by FPS.
2. Validate it over positive and negative Rear, Dwell, Occlusion, and Silent
Signal corpus videos, including FPS 15/20 and multiple runs.
3. Add the optional weapon stream adapter only if the feature is enabled in the
deployment, then separately validate its model/association parity.
4. Measure on the actual Pi-to-Vision and Spring Boot network path, including
GPU contention, authentication, retry behavior, TTL/cleanup, and fallback
deduplication.
