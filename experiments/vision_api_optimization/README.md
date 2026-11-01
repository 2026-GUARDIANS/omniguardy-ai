# Vision API optimization benchmark

`benchmark_http.py` launches the existing API without modifying it, then compares its default configuration with all three opt-in optimization flags enabled. It forces `VISION_SILENT_SIGNAL_SAME_HAND_ONLY=false` and `VISION_SILENT_SIGNAL_OPERATOR_ONLY=false`.

```powershell
.\.venv\Scripts\python.exe experiments\vision_api_optimization\benchmark_http.py --output experiment_results\vision_api_optimization\http_representative
```

Artifacts include raw HTTP responses, response/event/schema equality signatures, server timing logs, and best-effort CPU/RSS/GPU samples. The existing API timing logs supply per-stage values for tracking, occlusion, silent signal, rear approach, frame dimensions, dwell, and analysis wall time. Upload/storage and JSON serialization are intentionally reported only in HTTP wall time because the production endpoint has no independent timing hooks for those stages.
