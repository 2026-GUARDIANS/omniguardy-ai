"""Read-only Silent Signal prefix-equivalence and accuracy experiment.

This does not import or mutate the FastAPI endpoint.  It replays the current
detector state machine one frame at a time and stops on its first ``True``.
The rule constants and classifier are imported from the protected detector
modules, so this verifies early-confirmation semantics without proposing a
new threshold or classifier.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import cv2
import mediapipe as mp

ROOT = Path(__file__).resolve().parents[2]
HAND = ROOT / "hand_motion_desktop" / "src"
sys.path[:0] = [str(ROOT), str(HAND)]

from detector import (MAX_GAP_BETWEEN_REPS_FRAMES, MAX_TRANSITION_GAP_FRAMES,
                      MIN_FIST_FRAMES, REQUIRED_REPEATS, _build_detector,
                      analyze_video_for_silent_signal)
from detector_front import (FRONT_MAX_TRANSITION_GAP_FRAMES,
                            analyze_video_for_silent_signal_front)
from gesture_rules import RepeatedFistToOpenTracker, classify_gesture, frame_gesture_from_hands
from gesture_rules_front import classify_gesture_front

CASES = {
    "positive_175204": ("approach_signal/20261019_175204.mp4", "positive (user-provided)"),
    "positive_175021": ("approach_signal/20261019_175021.mp4", "positive/ambiguous (user-provided)"),
    "negative_174439": ("static_two_people/20261019_174439.mp4", "human-reviewed negative"),
    "negative_174454": ("static_two_people/20261019_174454.mp4", "human-reviewed negative"),
    "no_signal_174341": ("static_two_people/20261019_174341.mp4", "no-signal comparator"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def first_confirmation(path: Path, *, front: bool) -> dict[str, object]:
    """Exact current state machine, except return at its first confirmation."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(path)
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    detector = _build_detector()
    tracker = RepeatedFistToOpenTracker(
        required_repeats=REQUIRED_REPEATS, min_fist_frames=MIN_FIST_FRAMES,
        max_transition_gap_frames=(FRONT_MAX_TRANSITION_GAP_FRAMES if front else MAX_TRANSITION_GAP_FRAMES),
        max_gap_between_reps_frames=MAX_GAP_BETWEEN_REPS_FRAMES,
    )
    index = hand_frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                return {"detected": False, "confirmedFrame": None, "framesRead": index,
                        "handPresentFrames": hand_frames, "fps": fps}
            rgb = cv2.cvtColor(cv2.flip(frame, 1) if front else frame, cv2.COLOR_BGR2RGB)
            result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), int(index * 1000 / fps))
            gestures = []
            if result.hand_landmarks:
                hand_frames += 1
                for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                    label = handedness[0].category_name
                    gestures.append((classify_gesture_front if front else classify_gesture)(landmarks, label))
            if tracker.update(frame_gesture_from_hands(gestures)):
                return {"detected": True, "confirmedFrame": index, "framesRead": index + 1,
                        "handPresentFrames": hand_frames, "fps": fps}
            index += 1
    finally:
        capture.release()
        detector.close()


def full(path: Path, *, front: bool) -> dict[str, object]:
    result = analyze_video_for_silent_signal_front(path) if front else analyze_video_for_silent_signal(path)
    return {"detected": result.detected, "detectedFrame": result.detected_frame,
            "framesAnalyzed": result.frames_analyzed, "handPresentFrames": result.hand_present_frames}


def main() -> None:
    output = ROOT / "experiment_results" / "vision_api_optimization" / "silent_signal_prefix_accuracy"
    output.mkdir(parents=True, exist_ok=False)
    previous_same_hand = os.environ.get("VISION_SILENT_SIGNAL_SAME_HAND_ONLY")
    previous_operator = os.environ.get("VISION_SILENT_SIGNAL_OPERATOR_ONLY")
    os.environ["VISION_SILENT_SIGNAL_SAME_HAND_ONLY"] = "false"
    os.environ["VISION_SILENT_SIGNAL_OPERATOR_ONLY"] = "false"
    rows: list[dict[str, object]] = []
    try:
        for case, (relative, label) in CASES.items():
            video = ROOT / "final_demo" / "videos" / "03_keypad" / relative
            for mode, is_front in (("api_default_legacy", False), ("front_current_rule", True)):
                prefix, complete = first_confirmation(video, front=is_front), full(video, front=is_front)
                rows.append({"case": case, "label": label, "mode": mode, "video": str(video.relative_to(ROOT)),
                             "sha256": sha256(video), **prefix, **{f"full_{k}": v for k, v in complete.items()},
                             "prefixEqualsFullFirstEvent": (prefix["detected"] == complete["detected"] and
                                prefix["confirmedFrame"] == complete["detectedFrame"])})
    finally:
        if previous_same_hand is None: os.environ.pop("VISION_SILENT_SIGNAL_SAME_HAND_ONLY", None)
        else: os.environ["VISION_SILENT_SIGNAL_SAME_HAND_ONLY"] = previous_same_hand
        if previous_operator is None: os.environ.pop("VISION_SILENT_SIGNAL_OPERATOR_ONLY", None)
        else: os.environ["VISION_SILENT_SIGNAL_OPERATOR_ONLY"] = previous_operator
    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output / "provenance.json").write_text(json.dumps({
        "purpose": "read-only prefix/full equivalence; not HTTP or performance validation",
        "flags": {"VISION_SILENT_SIGNAL_SAME_HAND_ONLY": False, "VISION_SILENT_SIGNAL_OPERATOR_ONLY": False},
        "rules": {"MIN_FIST_FRAMES": MIN_FIST_FRAMES, "REQUIRED_REPEATS": REQUIRED_REPEATS,
                  "legacyMaxTransitionGap": MAX_TRANSITION_GAP_FRAMES,
                  "frontMaxTransitionGap": FRONT_MAX_TRANSITION_GAP_FRAMES},
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
