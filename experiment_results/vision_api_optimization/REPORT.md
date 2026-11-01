# MP4 Vision API optimization validation (2026-11-02)

## Scope and preservation

- `api/vision_api.py`, tracking/detector/model code, final-demo files, Streaming PoC, and Early Event PoC were not edited.
- The original working tree was already dirty: `api/vision_api.py` and `hand_motion_desktop/src/detector_front.py` were modified; numerous evaluation/experiment artifacts were untracked. No reset, restore, clean, commit, or push was run.
- New files are limited to `experiments/vision_api_optimization/` and this results directory.

## Implementation state

The three optimization flags are implemented only in the existing uncommitted `api/vision_api.py` diff and are opt-in (`false` when absent):

| Flag | Actual effect | Default |
| --- | --- | --- |
| `VISION_EXHIBITION_LIMIT_TRACKING_TO_WINDOW` | Stops AUDIO YOLO/BoT-SORT at `round(min(10s, duration) * fps)` using `LimitedVideoSource`; dwell still uses the same first 10-second window. | off |
| `VISION_EXHIBITION_PARALLEL_ANALYSIS` | Runs AUDIO OpenCV occlusion and the one YOLO/BoT-SORT call in two threads; no concurrent use of YOLO. | off |
| `VISION_KEYPAD_PARALLEL_ANALYSIS` | Runs Silent Signal (separate MediaPipe/OpenCV instance) and the one YOLO/BoT-SORT call in two threads. Rear/weapon rules remain after tracking. | off |

`VISION_SILENT_SIGNAL_SAME_HAND_ONLY` and `VISION_SILENT_SIGNAL_OPERATOR_ONLY` were explicitly set to `false` for every measurement.

## Representative real HTTP comparison

Machine: Windows 11, Python 3.14.3, Torch 2.13.0+cu130, RTX 4060 8 GiB. Both requests used a fresh API process per configuration, `VISION_AUDIO_MODE=exhibition_rules`, and the same uploaded source files.

| Video / trigger | Default HTTP wall | All relevant flags on | Change | HTTP / contract parity |
| --- | ---: | ---: | ---: | --- |
| `01_camera_occlusion/20261019_173225.mp4` / AUDIO | 39.235 s | 21.856 s | -44.3% (1.79x) | 200 / identical schema, event payloads, and `trackedPersonCount=2` |
| `03_keypad/approach_signal/20261019_175204.mp4` / KEYPAD | 48.417 s | 28.711 s | -40.7% (1.69x) | 200 / identical schema, event payloads, and `trackedPersonCount=3` |

This reproduces the direction and approximate size of the earlier results: AUDIO 43.813→23.980 s (-45.3%) and KEYPAD 51.044→31.828 s (-37.6%). Raw response bodies and equality signatures are in `http_representative_final/`.

## Stage profiling

| Stage | AUDIO default | AUDIO optimized | KEYPAD default | KEYPAD optimized |
| --- | ---: | ---: | ---: | ---: |
| Upload + metadata | HTTP-only; metadata 0.032 s | HTTP-only; metadata 0.024 s | HTTP-only; metadata 0.015 s | HTTP-only; metadata 0.021 s |
| OpenCV / occlusion | 11.040 s | 11.035 s (overlapped) | n/a | n/a |
| YOLO + BoT-SORT | 27.993 s | 21.686 s (10 s bound, overlapped) | 31.940 s | 28.499 s (overlapped) |
| HandLandmarker / Silent Signal | n/a | n/a | 16.259 s | 16.011 s (overlapped) |
| Dwell / rear rules | 0.003 s | 0.002 s | rear 0.021 s | rear 0.014 s |
| Analysis wall | 39.037 s | 21.691 s | 48.233 s | 28.529 s |

The current endpoint has no separate hooks for multipart storage, decoder-only time, event merge, or JSON serialization; each is contained in HTTP wall time. GPU samples showed roughly 819/824 MiB for default/optimized AUDIO and 818/982 MiB for default/optimized KEYPAD. The first benchmark's per-process CPU/RSS samples targeted the Windows launcher rather than all descendants, so they must not be interpreted; the benchmark was corrected to aggregate the server process tree for subsequent runs.

## Equivalence and risk

Within the fresh ON/OFF pair, AUDIO event types, frames, details, `trackedPersonCount`, status/result/visionEvents structure, and HTTP status matched exactly. The KEYPAD pair also matched exactly, but both new baseline and optimized responses had no events. This conflicts with the older stored run for the same clip, which had SILENT_SIGNAL at 114 and REAR_CLOSE_APPROACH_SUSPECTED at 44. It is a baseline reproducibility difference (not caused by the optimization), likely dependent on detector/tracker/runtime state; do not claim event-regression clearance for KEYPAD until this is investigated.

The unit regression `evaluation.test_exhibition_audio_optimizations` passed (4 tests), validating default sequential/unbounded behavior and independent opt-in behavior. Full 49 AUDIO + 15 KEYPAD replay was not run: the representative KEYPAD baseline discrepancy should be resolved before spending the long batch budget, because it would not establish useful optimization parity.

Error/environment handling was inspected in the endpoint: absent optimization variables use `false`; a non-`exhibition_rules`/`temporal_trial6` `VISION_AUDIO_MODE` raises HTTP 422; and `VISION_SILENT_SIGNAL_OPERATOR_ONLY=true` without `SILENT_SIGNAL_COORDINATE_MODE=front` raises HTTP 422. An attempted in-process HTTP assertion could not run because this virtual environment lacks the `httpx`/`httpx2` dependency required by its Starlette TestClient. The real benchmark used `requests` against Uvicorn and verified normal 200 responses; the two 422 paths remain a code-level, not end-to-end, verification.

## Findings and recommendation

1. The main bottlenecks are YOLO/BoT-SORT and, for KEYPAD, HandLandmarker. Event merge/rules are negligible.
2. AUDIO gains combine safe work removal (tracking beyond the already-used 10-second rule window) and overlap of independent CPU occlusion with GPU tracking. It has the clearest demonstration-safe case, but needs a broader AUDIO parity batch before permanent enablement.
3. KEYPAD parallelism materially reduces wall time, but creates CPU decoder contention and raised observed GPU memory to about 982 MiB. No shared YOLO tracker access occurs by design. Enable only after the current baseline event discrepancy is reproduced/resolved across the 15 clips.
4. Do not reduce FPS or resolution: those changes can alter Rear Approach, Silent Signal, and AUDIO occlusion semantics.
5. No production code change is required to obtain the measured best result: set all three opt-in flags to `true` before API process startup, leaving both Silent Signal experiment flags false. Current best measured representative performance is 21.856 s AUDIO and 28.711 s KEYPAD.
6. For a production-quality profiling decision, add non-response diagnostic timers around upload write, decode, event merge, serialization, and aggregate process-tree CPU/RSS collection. Keep response JSON unchanged.
