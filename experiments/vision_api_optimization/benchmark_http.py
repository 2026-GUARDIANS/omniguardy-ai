"""Isolated HTTP benchmark for the MP4 Vision API.

This starts an unmodified ``api.vision_api:app`` in a child process for every
configuration.  It records raw responses, schema/event equality, server timing
logs, and best-effort process/GPU samples.  It deliberately forces the two
Silent Signal experiment switches off.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import psutil
import requests

ROOT = Path(__file__).resolve().parents[2]
CASES = {
    "audio_occlusion": (ROOT / "final_demo/videos/01_camera_occlusion/20261019_173225.mp4", "AUDIO"),
    "keypad_compound": (ROOT / "final_demo/videos/03_keypad/approach_signal/20261019_175204.mp4", "KEYPAD"),
}
CONFIGS = {
    "baseline": {},
    "optimized": {
        "VISION_EXHIBITION_LIMIT_TRACKING_TO_WINDOW": "true",
        "VISION_EXHIBITION_PARALLEL_ANALYSIS": "true",
        "VISION_KEYPAD_PARALLEL_ANALYSIS": "true",
    },
}


def canonical(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        return {"notJson": payload}
    result = payload.get("result")
    if not isinstance(result, dict):
        return {"topKeys": sorted(payload), "result": result}
    return {
        "topKeys": sorted(payload), "resultKeys": sorted(result),
        "status": payload.get("status"), "resultStatus": result.get("status"),
        "visionEvents": result.get("visionEvents"),
        "trackedPersonCount": result.get("observations", {}).get("trackedPersonCount"),
    }


def sampler(pid: int, samples: list[dict[str, object]], stop: threading.Event) -> None:
    process = psutil.Process(pid)
    while not stop.wait(0.25):
        try:
            # Windows/Python can leave a lightweight launcher as the direct
            # Popen process while uvicorn's serving process is a descendant.
            # Aggregate the whole request-server tree rather than attributing
            # only the launcher RSS/CPU to inference.
            tree = [process, *process.children(recursive=True)]
            item: dict[str, object] = {"t": time.time(),
                "rss": sum(p.memory_info().rss for p in tree),
                "cpu": sum(p.cpu_percent(None) for p in tree), "processCount": len(tree)}
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                 timeout=3, check=False)
            if gpu.returncode == 0 and gpu.stdout.strip():
                util, memory = gpu.stdout.strip().splitlines()[0].split(",")
                item.update(gpuUtilPercent=float(util), gpuMemoryMiB=float(memory))
            samples.append(item)
        except (psutil.Error, OSError, subprocess.SubprocessError, ValueError):
            pass


def wait_ready(port: int, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("server stopped before readiness")
        try:
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise TimeoutError("server readiness timeout")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--configs", nargs="+", choices=CONFIGS, default=["baseline", "optimized"])
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, object]] = []
    reference: dict[str, object] = {}
    for config_name in args.configs:
        config_dir = args.output / config_name
        config_dir.mkdir()
        env = os.environ.copy()
        env.update({"PYTHONPATH": str(ROOT), "VISION_AUDIO_MODE": "exhibition_rules",
                    "VISION_SILENT_SIGNAL_SAME_HAND_ONLY": "false",
                    "VISION_SILENT_SIGNAL_OPERATOR_ONLY": "false",
                    "VISION_EXHIBITION_TIMING_LOG": "true", "VISION_KEYPAD_TIMING_LOG": "true"})
        for name in (*CONFIGS["optimized"],):
            env.pop(name, None)
        env.update(CONFIGS[config_name])
        with (config_dir / "server.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.vision_api:app",
                                        "--host", "127.0.0.1", "--port", str(args.port), "--log-level", "info"],
                                       cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
            try:
                wait_ready(args.port, process)
                for case in args.cases:
                    video, trigger_type = CASES[case]
                    if not video.is_file():
                        raise FileNotFoundError(video)
                    samples: list[dict[str, object]] = []
                    stop = threading.Event()
                    thread = threading.Thread(target=sampler, args=(process.pid, samples, stop), daemon=True)
                    thread.start()
                    started = time.perf_counter()
                    with video.open("rb") as source:
                        response = requests.post(f"http://127.0.0.1:{args.port}/analyze/vision",
                            files={"file": (video.name, source, "video/mp4")},
                            data={"triggerId": f"opt-{config_name}-{case}", "triggerType": trigger_type,
                                  "securityEventId": f"opt-{config_name}-{case}",
                                  "triggeredAt": datetime.now(timezone.utc).isoformat()}, timeout=900)
                    wall = time.perf_counter() - started
                    stop.set(); thread.join(timeout=3)
                    try: payload: object = response.json()
                    except ValueError: payload = {"rawText": response.text}
                    (config_dir / f"{case}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                    (config_dir / f"{case}_resources.json").write_text(json.dumps(samples, indent=2), encoding="utf-8")
                    signature = canonical(payload)
                    if config_name == "baseline": reference[case] = signature
                    peak_rss = max((int(x.get("rss", 0)) for x in samples), default=0)
                    peak_gpu = max((float(x.get("gpuMemoryMiB", 0)) for x in samples), default=0)
                    rows.append({"config": config_name, "case": case, "httpStatus": response.status_code,
                                 "wallSeconds": round(wall, 3), "schemaAndEventsEqualBaseline": signature == reference.get(case),
                                 "peakProcessRssMiB": round(peak_rss / 1048576, 1), "peakGpuMemoryMiB": peak_gpu,
                                 "sampleCount": len(samples), "signature": json.dumps(signature, ensure_ascii=False, sort_keys=True)})
                    print(f"{config_name}/{case}: HTTP {response.status_code}, {wall:.3f}s")
            finally:
                process.terminate()
                try: process.wait(timeout=20)
                except subprocess.TimeoutExpired: process.kill()
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (args.output / "environment.json").write_text(json.dumps({"python": sys.version, "configs": CONFIGS,
        "silentSignalFlagsForcedOff": True}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
