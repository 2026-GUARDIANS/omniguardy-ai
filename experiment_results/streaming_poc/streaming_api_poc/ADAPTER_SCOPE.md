# Phase B adapter scope — implemented initial adapter

The following read-only reuse targets are now connected by a session-local
adapter. Existing production functions are not copied or modified:

| Domain | Existing code | Why an adapter is required |
|---|---|---|
| YOLO tracking | `api/vision_api.py:InferenceService._tracking_features()` | Current input is `VideoCapture(video_path)` / full source, while streaming requires per-session persistent tracker state and ordered JPEG frames. |
| KEYPAD Rear Approach | `api/vision_api.py:analyze_rear_approach()` → `tracking/keypad_rules_front.py:detect_front_rear_approach()` | Rule consumes a complete tracking DataFrame; online adapter must retain the equivalent per-frame records and run the unchanged rule at finalization. |
| AUDIO occlusion | `final_demo/src/exhibition_audio_service.py:detect_camera_occlusion()` | Current rule takes a video path and fps; a frame-state adapter must be parity-tested at 1920x1080 before online use. |
| AUDIO dwell | `final_demo/src/exhibition_audio_service.py:detect_person_dwell()` | Depends on complete tracking records/time interpretation. |

The adapter maintains a session-local persistent YOLO/BoT-SORT worker and an
append-only tracking record compatible with the existing DataFrame schema;
after `/vision/complete` queue drain, it calls existing final rules over that
record. Initial parity is reported in `phase_b/PHASE_B_REPORT.md`; it is not
full-corpus parity.

O1 ownership logic remains excluded: it is an experimental rule with a raw
`175021` miss and a remaining `174439` false positive.
