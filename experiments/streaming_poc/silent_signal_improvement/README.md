# Silent Signal improvement experiment

`run_experiment.py` is an isolated online-state-machine experiment. It imports
the existing MediaPipe construction and gesture classifier but never edits or
monkey-patches them.

```powershell
.\.venv\Scripts\python.exe experiments/streaming_poc/silent_signal_improvement/run_experiment.py --case candidate_175204
```

Omit `--case` to run all seven cases. Each frame is resized to 960x540 before
JPEG-equivalent online inference; the source frame index and MediaPipe video
timestamp still use the original 30 FPS stream. Results are placed only under
`experiment_results/streaming_poc/silent_signal_improvement/`.

The three experimental modes are diagnostic only. They are not environment
flags, do not alter `detector_front.py`, and do not send callbacks.
