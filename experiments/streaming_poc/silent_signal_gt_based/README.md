# GT-based ownership PoC

Run one isolated condition:

```powershell
.\.venv\Scripts\python.exe experiments/streaming_poc/silent_signal_gt_based/run_gt_poc.py --case positive_175021 --mode jpeg
```

`raw` is original frame input; `jpeg` is 960x540 JPEG quality 85 decode input.
The script uses existing stored full64 frame boxes only for offline ownership
analysis. It does not run a callback, modify an API, or modify a production
rule. Results are written only to `experiment_results/streaming_poc/silent_signal_gt_based/`.
