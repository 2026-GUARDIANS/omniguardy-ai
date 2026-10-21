"""Post selected read-only demo videos to an already-running local API.

The program does not start a server, mutate input videos, or run models on
its own.  It records actual HTTP response bodies and wall times in a new
output directory supplied by the caller.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
import time

import requests


ROOT = Path(__file__).resolve().parents[2]
VIDEOS = ROOT / "final_demo" / "videos"

CASES = {
    "audio_dwell": ("02_knock/exit_reentry/20261019_173843.mp4", "AUDIO"),
    "audio_occlusion": ("01_camera_occlusion/20261019_173225.mp4", "AUDIO"),
    "audio_no_event": ("02_knock/peeking/20261019_173126.mp4", "AUDIO"),
    "keypad": ("03_keypad/approach_signal/20261019_175204.mp4", "KEYPAD"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8011/analyze/vision")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cases", nargs="+", choices=sorted(CASES), required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = []
    for case in args.cases:
        relative_path, trigger_type = CASES[case]
        video = VIDEOS / relative_path
        started = time.perf_counter()
        with video.open("rb") as stream:
            response = requests.post(
                args.url,
                files={"file": (video.name, stream, "video/mp4")},
                data={
                    "triggerId": f"integration-{case}",
                    "triggerType": trigger_type,
                    "securityEventId": f"security-{case}",
                    "triggeredAt": datetime.now(timezone.utc).isoformat(),
                },
                timeout=600,
            )
        elapsed = round(time.perf_counter() - started, 3)
        try:
            body = response.json()
        except ValueError:
            body = {"rawText": response.text}
        (args.output / f"{case}_response.json").write_text(
            json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        result = body.get("result", {}) if isinstance(body, dict) else {}
        summary.append({
            "case": case,
            "relative_path": relative_path,
            "trigger_type": trigger_type,
            "http_status": response.status_code,
            "wall_seconds": elapsed,
            "analysis_mode": result.get("observations", {}).get("analysisMode", ""),
            "behavior_is_null": result.get("behavior") is None if result else "",
            "event_types": "|".join(event.get("eventType", "") for event in result.get("visionEvents", [])),
            "security_event_id": result.get("trigger", {}).get("securityEventId", ""),
        })
    with (args.output / "http_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
