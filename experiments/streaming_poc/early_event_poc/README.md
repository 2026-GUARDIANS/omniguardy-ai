# Early Silent Signal callback PoC

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe experiments/streaming_poc/early_event_poc/early_signal_callback_poc.py
```

The script starts an ephemeral `127.0.0.1` mock HTTP receiver, replays `175204`
at source FPS, feeds JPEG bytes to independent MediaPipe and persistent YOLO
workers, and emits an EARLY_EVENT at the existing default-rule confirmation
frame. It deliberately makes the first callback return HTTP 500 and retries
once to verify idempotency fields. Callback transport is independent from the
MediaPipe worker. The script refuses callback URLs except `http://127.0.0.1:<port>`;
`VISION_EARLY_CALLBACK_URL` is only for a locally started test receiver.

Results are written to `experiment_results/streaming_poc/early_event/`.

This PoC imports the existing detector helpers but does not change them. It
does not implement an API endpoint, does not contact Spring Boot, and does not
claim that current Silent Signal accuracy is adequate for early user alerts.
