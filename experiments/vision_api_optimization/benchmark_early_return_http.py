"""Real HTTP timing for KEYPAD early return, split by coordinate mode."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import requests

ROOT = Path(__file__).resolve().parents[2]
CASES = {
    "175204": "approach_signal/20261019_175204.mp4",
    "175021": "approach_signal/20261019_175021.mp4",
    "174439": "static_two_people/20261019_174439.mp4",
    "174454": "static_two_people/20261019_174454.mp4",
    "174341": "static_two_people/20261019_174341.mp4",
}
CONFIGS = {
    "legacy_full_parallel": {"SILENT_SIGNAL_COORDINATE_MODE": "legacy", "VISION_KEYPAD_PARALLEL_ANALYSIS": "true"},
    "legacy_early": {"SILENT_SIGNAL_COORDINATE_MODE": "legacy", "VISION_KEYPAD_EARLY_RETURN_ON_SILENT_SIGNAL": "true"},
    "front_full_parallel": {"SILENT_SIGNAL_COORDINATE_MODE": "front", "VISION_KEYPAD_PARALLEL_ANALYSIS": "true"},
    "front_early": {"SILENT_SIGNAL_COORDINATE_MODE": "front", "VISION_KEYPAD_EARLY_RETURN_ON_SILENT_SIGNAL": "true"},
}


def ready(port: int, proc: subprocess.Popen[str]) -> None:
    until = time.monotonic() + 90
    while time.monotonic() < until:
        if proc.poll() is not None: raise RuntimeError("server exited before ready")
        try:
            if requests.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200: return
        except requests.RequestException: pass
        time.sleep(.25)
    raise TimeoutError("server readiness")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=18082); args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False); rows = []
    for config, values in CONFIGS.items():
        directory = args.output / config; directory.mkdir()
        env = os.environ.copy(); env.update({"PYTHONPATH": str(ROOT), "VISION_SILENT_SIGNAL_SAME_HAND_ONLY": "false",
            "VISION_SILENT_SIGNAL_OPERATOR_ONLY": "false", "VISION_KEYPAD_EARLY_RETURN_ON_SILENT_SIGNAL": "false",
            "VISION_KEYPAD_PARALLEL_ANALYSIS": "false", **values})
        with (directory / "server.log").open("w", encoding="utf-8") as log:
            proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "api.vision_api:app", "--host", "127.0.0.1",
                "--port", str(args.port), "--log-level", "info"], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
            try:
                ready(args.port, proc)
                for name, relative in CASES.items():
                    video = ROOT / "final_demo" / "videos" / "03_keypad" / relative
                    start = time.perf_counter()
                    with video.open("rb") as source:
                        response = requests.post(f"http://127.0.0.1:{args.port}/analyze/vision", files={"file": (video.name, source, "video/mp4")},
                            data={"triggerId": f"early-{config}-{name}", "triggerType": "KEYPAD", "securityEventId": f"early-{name}",
                                  "triggeredAt": datetime.now(timezone.utc).isoformat()}, timeout=900)
                    wall = time.perf_counter() - start
                    payload = response.json(); (directory / f"{name}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                    result = payload.get("result", {}); events = result.get("visionEvents", [])
                    silent = next((e for e in events if e.get("eventType") == "SILENT_SIGNAL"), None)
                    rows.append({"config": config, "case": name, "httpStatus": response.status_code, "httpWallSeconds": round(wall, 3),
                        "analysisComplete": result.get("analysisComplete", True), "earlyReturnReason": result.get("earlyReturnReason"),
                        "silentDetectedFrame": silent.get("detectedFrame") if silent else None,
                        "trackedPersonCount": result.get("observations", {}).get("trackedPersonCount"),
                        "eventTypes": ";".join(e.get("eventType", "") for e in events)})
                    print(config, name, response.status_code, round(wall, 3))
            finally:
                proc.terminate()
                try: proc.wait(timeout=20)
                except subprocess.TimeoutExpired: proc.kill()
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__": main()
