# KEYPAD Silent Signal early HTTP return

## Implementation

`VISION_KEYPAD_EARLY_RETURN_ON_SILENT_SIGNAL` is a new opt-in flag; absence is `false`. It is considered only in the KEYPAD branch. AUDIO and the `false` KEYPAD path retain their existing flow.

With the flag on, `api/silent_signal_early.py` reuses the production HandLandmarker builder, existing classifier, constants, and `RepeatedFistToOpenTracker`, preserving both `legacy` and `front` coordinate modes. It stops only after the state machine's first `True` result. It does not change a threshold, classifier, or ownership experiment.

For a detected signal, the endpoint returns before constructing `InferenceService` or submitting YOLO. This deliberately avoids unsafe Python-thread/CUDA cancellation: there is no live YOLO task, no tracker state, no GPU allocation, and no worker that can retain the upload temp file after the response. OpenCV and the HandLandmarker are closed before the return. The endpoint `finally` then deletes the temporary upload normally.

If there is no signal, prefix replay reaches EOF and its full MediaPipe result is passed to `_analyze_keypad` as `precomputed_signal`; MediaPipe is not run twice, then normal tracking/rear/weapon analysis runs. This is safe but does not overlap MediaPipe and YOLO, so it is slower than the pre-existing parallel path for no-signal input.

Front same-hand and operator-only filters are intentionally not approximated: early mode logs a disablement and uses the full existing path for those combinations because they need their established ownership/tracking semantics.

## Wire contract

The response remains `status/result` with the existing trigger, video, `behavior:null`, `visionEvents`, and `observations` objects. An early response adds:

```json
"analysisComplete": false,
"earlyReturnReason": "SILENT_SIGNAL"
```

It emits the same Silent Signal event metadata as full analysis. `trackedPersonCount` is `null`, never invented as zero; `hasTracking:false` means tracking was deliberately not run. Rear/weapon events are absent because they are **not yet analyzed**, not because they were judged negative. A strict backend that requires `trackedPersonCount` to be an integer must be made nullable or reject early mode; enabling it without that contract check is not recommended.

## Real HTTP benchmark

Four fresh Uvicorn processes were used: legacy/front × full-parallel/early. Both experiment flags were false; full used the existing `VISION_KEYPAD_PARALLEL_ANALYSIS=true`. Times are client request-start through response receipt.

| Mode / clip | Full parallel | Early | Confirmed frame | Early result |
| --- | ---: | ---: | ---: | --- |
| legacy 175204 | 28.387 s | 43.638 s | none | full fallback |
| legacy 175021 | 27.629 s | 20.759 s | 345 | early, -6.870 s |
| legacy 174439 (negative) | 26.407 s | 1.968 s | 31 | false positive returned early |
| legacy 174454 (negative) | 28.295 s | 4.135 s | 100 | false positive returned early |
| legacy 174341 (no signal) | 27.859 s | 42.732 s | none | full fallback |
| front 175204 | 28.416 s | 5.483 s | 114 | early, -22.933 s (80.7%) |
| front 175021 | 27.710 s | 20.882 s | 333 | early, -6.828 s |
| front 174439 (negative) | 26.483 s | 2.348 s | 35 | false positive returned early |
| front 174454 (negative) | 29.946 s | 9.181 s | 174 | false positive returned early |
| front 174341 (no signal) | 30.400 s | 45.923 s | none | full fallback |

For front 175204, server timing logged confirmation/cleanup at 5.374 s and Uvicorn logged the HTTP 200 immediately after; the client received it at 5.483 s. No `[vision_api] YOLO device=...` log appeared for any early-detected request. The only YOLO initialization in each early server was associated with a preceding/following no-signal fallback request. Thus this verifies actual early HTTP completion, rather than merely early event detection.

The full versus early event parity for Silent Signal holds for every detected row and coordinate mode. Expected multi-event difference: full `175021` additionally returned `REAR_CLOSE_APPROACH_SUSPECTED`, whereas early response explicitly declares incomplete analysis.

Raw payloads, summary, and server logs: [early_return_http](early_return_http/).

## Validation and recommendation

- The feature default is **off**. With `VISION_KEYPAD_EARLY_RETURN_ON_SILENT_SIGNAL=false`, and for every AUDIO request, the new early-prefix path is not entered; existing AUDIO and complete KEYPAD analysis behavior is retained.
- The coordinate mode is never changed by this feature. It evaluates the configured `legacy` or `front` mode, and falls back to the existing complete path for unsupported coordinate/filter combinations.
- `experiments.vision_api_optimization.test_early_return_contract` and the prior AUDIO optimization regression ran successfully: 6 tests total.
- Existing `VISION_KEYPAD_EARLY_RETURN_ON_SILENT_SIGNAL=false` behavior is unchanged by unit verification. AUDIO did not enter the new branch.
- The option should be enabled only after backend validation that nullable `trackedPersonCount` and `analysisComplete=false` are accepted. For front-mode demo `175204` it provides a large, measured response gain. It will also accelerate the known false positives; it is a latency feature, not an accuracy improvement.
- Do not enable it expecting a no-signal performance gain. The safe no-background-worker design costs approximately 12–15 seconds versus the existing parallel full path on this machine.
