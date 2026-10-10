"""Compare serving A18 landmark extraction with the frozen offline diagnosis."""
from __future__ import annotations

import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracking.handle_roi_rules import HandleRoi, HandleRuleConfig, analyze_handle_motion


REFERENCE = ROOT / "test" / "final_test" / "output" / "a18_handle_roi_diagnosis" / "clip_features.csv"
OUT = ROOT / "evaluation" / "a18_rule_validation_20261011"
POSE = ROOT / "models" / "pose_landmarker_lite.task"


def main() -> None:
    config = HandleRuleConfig(HandleRoi(.35, .44, .57, .68), 3, 2, 8., .05)
    with REFERENCE.open(encoding="utf-8-sig") as handle:
        reference = list(csv.DictReader(handle))
    rows = []
    for item in reference:
        path = ROOT / item["clip"]
        result = analyze_handle_motion(path, POSE, 10, config)
        expected = item["rule_a18_moving_candidate"] == "True"
        row = {
            "clip": item["clip"], "actual_class": item["actual_class"],
            "expected": expected, "serving_detected": result.detected,
            "match": expected == result.detected,
            "serving_roi_samples": result.roi_samples,
            "serving_max_consecutive": result.max_consecutive_samples,
            "serving_angle_range": result.elbow_angle_range,
            "serving_path_length": result.wrist_path_length,
        }
        rows.append(row)
        print(f"{item['clip']}: expected={expected} serving={result.detected}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "serving_vs_diagnostic.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    total = len(rows); matched = sum(row["match"] for row in rows)
    print(f"matched={matched}/{total}")
    if matched != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
