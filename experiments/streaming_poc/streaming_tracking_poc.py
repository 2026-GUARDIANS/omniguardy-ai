"""Phase 1 only: frame-at-a-time YOLO/BoT-SORT tracking PoC.

This deliberately does not start a FastAPI server, alter the existing API, or
write any video source.  It reads a recorded MP4 as a virtual camera and feeds
each decoded BGR frame directly to ``YOLO.track(..., persist=True)``.  The
stored full64 tracking records are used only as an offline comparison oracle.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import torch
import ultralytics
from ultralytics import YOLO
from ultralytics.utils import ROOT as ULTRALYTICS_ROOT


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIDEOS = PROJECT_ROOT / "final_demo" / "videos"
FULL64 = PROJECT_ROOT / "final_demo" / "results_front_rules" / "runs" / "full64_20261020" / "videos"
MODEL_PATH = PROJECT_ROOT / "tracking" / "yolov8n.pt"
RESULTS_ROOT = PROJECT_ROOT / "experiment_results" / "streaming_poc"

DEFAULT_CASES = {
    "approach_signal_175204": "03_keypad/approach_signal/20261019_175204.mp4",
    "approach_signal_174955": "03_keypad/approach_signal/20261019_174955.mp4",
}


@dataclass
class StreamingTrackingSession:
    """One isolated event/session and its persistent Ultralytics predictor."""

    event_id: str
    session_id: str
    trigger_type: str
    model: YOLO
    frame_index: int = -1
    finalized: bool = False

    def process(self, frame: Any, *, frame_index: int, captured_at_seconds: float) -> list[dict[str, Any]]:
        if self.finalized:
            raise RuntimeError("Frame received after session finalization")
        if frame_index <= self.frame_index:
            raise ValueError(f"Non-monotonic frameIndex {frame_index}; last={self.frame_index}")
        self.frame_index = frame_index
        device: int | str = 0 if torch.cuda.is_available() else "cpu"
        # persist=True is the property under test: the predictor's BoT-SORT
        # instance is reused for the next frame of this session.
        result = self.model.track(
            frame, tracker="botsort.yaml", persist=True, save=False, conf=0.3,
            classes=[0], stream=False, device=device, verbose=False, imgsz=640,
        )[0]
        records: list[dict[str, Any]] = []
        if result.boxes is not None:
            for box in result.boxes:
                if box.id is None:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                records.append({
                    "frame": frame_index,
                    "capturedAtSeconds": round(captured_at_seconds, 6),
                    "track_id": int(box.id[0]),
                    "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                    "confidence": float(box.conf[0]),
                })
        return records

    def finalize(self) -> dict[str, Any]:
        """Explicitly discard BoT-SORT state before the model is reused."""
        if self.finalized:
            return {"alreadyFinalized": True, "trackerCount": 0}
        predictor = getattr(self.model, "predictor", None)
        trackers = list(getattr(predictor, "trackers", []) or [])
        for tracker in trackers:
            tracker.reset()
        self.finalized = True
        return {"alreadyFinalized": False, "trackerCount": len(trackers)}


def find_full64(relative_path: str) -> dict[str, Any]:
    for candidate in FULL64.glob("*.json"):
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        if payload.get("relative_path") == relative_path:
            return payload
    raise FileNotFoundError(f"No full64 result for {relative_path}")


def iou(first: dict[str, Any], second: dict[str, Any]) -> float:
    left, top = max(first["x1"], second["x1"]), max(first["y1"], second["y1"])
    right, bottom = min(first["x2"], second["x2"]), min(first["y2"], second["y2"])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first["x2"] - first["x1"]) * max(0.0, first["y2"] - first["y1"])
    second_area = max(0.0, second["x2"] - second["x1"]) * max(0.0, second["y2"] - second["y1"])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def compare_tracking(baseline: list[dict[str, Any]], online: list[dict[str, Any]]) -> dict[str, Any]:
    by_frame_baseline: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_frame_online: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in baseline:
        by_frame_baseline[int(row["frame"])].append(row)
    for row in online:
        by_frame_online[int(row["frame"])].append(row)

    ious, id_equal, matched = [], 0, 0
    id_mapping: dict[tuple[int, int], int] = defaultdict(int)
    for frame in sorted(set(by_frame_baseline) | set(by_frame_online)):
        available = list(by_frame_online[frame])
        for reference in by_frame_baseline[frame]:
            if not available:
                continue
            candidate = max(available, key=lambda row: iou(reference, row))
            overlap = iou(reference, candidate)
            if overlap < 0.5:
                continue
            available.remove(candidate)
            matched += 1
            ious.append(overlap)
            same = int(reference["track_id"]) == int(candidate["track_id"])
            id_equal += int(same)
            id_mapping[(int(reference["track_id"]), int(candidate["track_id"]))] += 1

    mappings = [
        {"mp4TrackId": source, "onlineTrackId": target, "matchedFrames": count}
        for (source, target), count in sorted(id_mapping.items())
    ]
    return {
        "baselineBoxRecords": len(baseline),
        "onlineBoxRecords": len(online),
        "iouMatchedBoxesAt0_5": matched,
        "meanIouForMatchedBoxes": round(sum(ious) / len(ious), 5) if ious else None,
        "sameTrackIdAmongMatchedBoxes": id_equal,
        "sameTrackIdRateAmongMatchedBoxes": round(id_equal / matched, 5) if matched else None,
        "trackIdMapping": mappings,
    }


def continuity(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for track_id, group in (pd.DataFrame(records).groupby("track_id") if records else []):
        frames = sorted(int(value) for value in group["frame"])
        gaps = [later - earlier - 1 for earlier, later in zip(frames, frames[1:])]
        output.append({
            "trackId": int(track_id), "startFrame": frames[0], "endFrame": frames[-1],
            "observedFrames": len(frames), "maxInternalGapFrames": max(gaps, default=0),
        })
    return output


def run_video(case: str, relative_path: str, output: Path) -> dict[str, Any]:
    source = VIDEOS / relative_path
    stored = find_full64(relative_path)
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    session = StreamingTrackingSession(
        event_id=f"poc-{case}", session_id=f"poc-{case}-s1", trigger_type="KEYPAD", model=YOLO(str(MODEL_PATH)),
    )
    records: list[dict[str, Any]] = []
    index = 0
    started = time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            records.extend(session.process(frame, frame_index=index, captured_at_seconds=index / fps))
            index += 1
    finally:
        cap.release()
    elapsed = time.perf_counter() - started
    reset = session.finalize()
    comparison = compare_tracking(stored["full"].get("tracking", []), records)
    result = {
        "case": case, "relativePath": relative_path,
        "session": {"eventId": session.event_id, "sessionId": session.session_id, "triggerType": session.trigger_type},
        "source": {"fps": fps, "width": width, "height": height, "framesRead": index},
        "online": {"boxRecords": len(records), "wallSeconds": round(elapsed, 4), "processingFps": round(index / elapsed, 4)},
        "trackerFinalize": reset,
        "comparisonToStoredFull64": comparison,
        "onlineContinuity": continuity(records),
        "storedFull64Config": stored.get("config", {}),
        "note": "Full64 is a prior inference artifact, not human ground truth. This comparison tests tracking parity only.",
    }
    (output / f"{case}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(records).to_csv(output / f"{case}_online_tracking.csv", index=False, encoding="utf-8-sig")
    return result


def reset_smoke_test(relative_path: str) -> dict[str, Any]:
    """Prove a single model's tracker is reset between two event sessions."""
    cap = cv2.VideoCapture(str(VIDEOS / relative_path))
    ok, first = cap.read(); cap.release()
    if not ok:
        raise RuntimeError(relative_path)
    model = YOLO(str(MODEL_PATH))
    first_session = StreamingTrackingSession("reset-event-1", "reset-s1", "KEYPAD", model)
    before = first_session.process(first, frame_index=0, captured_at_seconds=0.0)
    finalize = first_session.finalize()
    second_session = StreamingTrackingSession("reset-event-2", "reset-s2", "KEYPAD", model)
    after = second_session.process(first, frame_index=0, captured_at_seconds=0.0)
    second_session.finalize()
    return {
        "firstSessionTrackIds": [row["track_id"] for row in before],
        "secondSessionTrackIdsAfterReset": [row["track_id"] for row in after],
        "firstSessionFinalize": finalize,
        "interpretation": "IDs restarting after reset is expected; session isolation is proven by explicit tracker reset, not globally unique IDs.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", choices=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=RESULTS_ROOT / "phase1")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    selected = args.case or list(DEFAULT_CASES)
    manifest = {
        "phase": "Phase 1 online tracker smoke/parity only",
        "ultralyticsVersion": ultralytics.__version__,
        "botsortYaml": str(ULTRALYTICS_ROOT / "cfg" / "trackers" / "botsort.yaml"),
        "torchCudaAvailable": torch.cuda.is_available(),
        "cases": [run_video(case, DEFAULT_CASES[case], output) for case in selected],
        "resetSmoke": reset_smoke_test(DEFAULT_CASES[selected[0]]),
        "notRun": ["JPEG encode/decode", "FPS/resize/backlog", "MP4 recording branch", "MediaPipe online", "AUDIO event parity", "HTTP streaming API"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "cases": selected}, ensure_ascii=False))


if __name__ == "__main__":
    main()
