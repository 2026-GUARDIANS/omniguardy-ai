"""Measure the existing MP4 Vision API without changing its response contract.

Run one server configuration at a time, for example::

  python evaluation/benchmark_exhibition_api.py --url http://127.0.0.1:8010/analyze/vision --output experiment_results/api_baseline

The three default videos cover an AUDIO occlusion positive, a dwell positive,
and a normal-wait negative.  The script stores raw response bodies so different
server configurations can be compared event-for-event after timing.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES: dict[str, tuple[Path, str]] = {
    "occlusion_positive": (ROOT / "final_demo/videos/01_camera_occlusion/20261019_173225.mp4", "AUDIO"),
    "dwell_positive": (ROOT / "final_demo/videos/02_knock/exit_reentry/20261019_173843.mp4", "AUDIO"),
    "normal_negative": (ROOT / "final_demo/videos/02_knock/normal_wait/20261019_141346.mp4", "AUDIO"),
    # KEYPAD cases cover a signal-only candidate, rear-only positive, compound
    # positive, two-person negative, and a multi-person sequence respectively.
    "keypad_signal_positive": (ROOT / "final_demo/videos/03_keypad/approach_signal/20261019_174955.mp4", "KEYPAD"),
    "keypad_rear_positive": (ROOT / "final_demo/videos/03_keypad/rear_approach/20261019_174606.mp4", "KEYPAD"),
    "keypad_compound_positive": (ROOT / "final_demo/videos/03_keypad/approach_signal/20261019_175204.mp4", "KEYPAD"),
    "keypad_negative": (ROOT / "final_demo/videos/03_keypad/static_two_people/20261019_174341.mp4", "KEYPAD"),
    "keypad_negative_fp_174439": (ROOT / "final_demo/videos/03_keypad/static_two_people/20261019_174439.mp4", "KEYPAD"),
    "keypad_negative_fp_174454": (ROOT / "final_demo/videos/03_keypad/static_two_people/20261019_174454.mp4", "KEYPAD"),
    "keypad_multi_person": (ROOT / "final_demo/videos/03_keypad/approach_signal/20261019_175021.mp4", "KEYPAD"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cases", nargs="+", choices=sorted(DEFAULT_CASES), default=sorted(DEFAULT_CASES))
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")
    args.output.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, object]] = []
    for run in range(1, args.runs + 1):
        for case in args.cases:
            video, trigger_type = DEFAULT_CASES[case]
            if not video.is_file():
                raise SystemExit(f"Missing benchmark video: {video}")
            started = time.perf_counter()
            with video.open("rb") as stream:
                response = requests.post(
                    args.url,
                    files={"file": (video.name, stream, "video/mp4")},
                    data={
                        "triggerId": f"api-benchmark-{case}-{run}",
                        "triggerType": trigger_type,
                        "securityEventId": f"api-benchmark-{case}-{run}",
                        "triggeredAt": datetime.now(timezone.utc).isoformat(),
                    },
                    timeout=args.timeout,
                )
            wall_seconds = time.perf_counter() - started
            try:
                payload = response.json()
            except ValueError:
                payload = {"rawText": response.text}
            (args.output / f"{run:02d}_{case}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            result = payload.get("result", {}) if isinstance(payload, dict) else {}
            rows.append({
                "run": run,
                "case": case,
                "video": str(video.relative_to(ROOT)),
                "httpStatus": response.status_code,
                "wallSeconds": round(wall_seconds, 3),
                "trackedPersonCount": result.get("observations", {}).get("trackedPersonCount"),
                "visionEvents": json.dumps(result.get("visionEvents", []), ensure_ascii=False, sort_keys=True),
            })
            print(f"{run}/{args.runs} {case}: HTTP {response.status_code}, {wall_seconds:.3f}s")

    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
