"""AUDIO-only A18 handle ROI post-processing helpers.

This module deliberately does not alter the Temporal model's features or the
public output of ``InferenceService._pose_features``.  Landmark extraction is
an optional second Pose pass, only after Trial 6 has produced A17 or A18.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
from typing import Callable, Iterable

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision


Log = Callable[[str], None]


@dataclass(frozen=True)
class HandleRoi:
    x0: float
    y0: float
    x1: float
    y1: float

    @classmethod
    def parse(cls, raw: str | None) -> "HandleRoi | None":
        if raw is None or not raw.strip():
            return None
        try:
            values = [float(value.strip()) for value in raw.split(",")]
        except ValueError:
            return None
        if len(values) != 4 or not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
            return None
        roi = cls(*values)
        return roi if roi.x0 < roi.x1 and roi.y0 < roi.y1 else None

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1


@dataclass(frozen=True)
class HandleRuleConfig:
    roi: HandleRoi
    min_roi_samples: int
    min_consecutive_samples: int
    min_elbow_angle_degrees: float
    min_wrist_path_length: float

    @classmethod
    def from_environment(cls, log: Log = print) -> "HandleRuleConfig | None":
        enabled = os.getenv("VISION_HANDLE_RULE", "0").strip()
        if enabled == "0":
            return None
        if enabled != "1":
            log(f"[vision_api] WARNING: VISION_HANDLE_RULE={enabled!r} is invalid; handle rule disabled")
            return None
        roi = HandleRoi.parse(os.getenv("VISION_HANDLE_ROI"))
        if roi is None:
            log("[vision_api] WARNING: VISION_HANDLE_ROI must be x0,y0,x1,y1 normalized with x0<x1 and y0<y1; handle rule disabled")
            return None
        try:
            min_samples = int(os.getenv("VISION_HANDLE_MIN_ROI_SAMPLES", "3"))
            min_consecutive = int(os.getenv("VISION_HANDLE_MIN_CONSECUTIVE_SAMPLES", "2"))
            min_angle = float(os.getenv("VISION_HANDLE_MIN_ELBOW_ANGLE_DEGREES", "8"))
            min_path = float(os.getenv("VISION_HANDLE_MIN_WRIST_PATH_LENGTH", "0.05"))
        except ValueError:
            log("[vision_api] WARNING: invalid VISION_HANDLE_* threshold; handle rule disabled")
            return None
        if (min_samples < 1 or min_consecutive < 1 or min_consecutive > min_samples
                or not math.isfinite(min_angle) or min_angle < 0
                or not math.isfinite(min_path) or min_path < 0):
            log("[vision_api] WARNING: invalid VISION_HANDLE_* threshold range; handle rule disabled")
            return None
        return cls(roi, min_samples, min_consecutive, min_angle, min_path)


@dataclass(frozen=True)
class WristSample:
    frame: int
    x: float
    y: float
    angle: float
    inside: bool


@dataclass(frozen=True)
class HandleRuleResult:
    detected: bool
    pose_samples: int
    roi_samples: int
    max_consecutive_samples: int
    elbow_angle_range: float
    wrist_path_length: float


def _angle(shoulder: object, elbow: object, wrist: object) -> float:
    a = np.array([shoulder.x - elbow.x, shoulder.y - elbow.y])
    b = np.array([wrist.x - elbow.x, wrist.y - elbow.y])
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return float("nan")
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / denominator, -1, 1))))


def evaluate_samples(samples: Iterable[WristSample], config: HandleRuleConfig) -> HandleRuleResult:
    """Use the identical aggregate calculations as the offline diagnosis."""
    values = list(samples)
    streak = longest = 0
    for sample in values:
        streak = streak + 1 if sample.inside else 0
        longest = max(longest, streak)
    roi_samples = [sample for sample in values if sample.inside]
    path_length = sum(
        math.hypot(second.x - first.x, second.y - first.y)
        for first, second in zip(roi_samples, roi_samples[1:])
    )
    angles = [sample.angle for sample in roi_samples if math.isfinite(sample.angle)]
    angle_range = max(angles) - min(angles) if angles else 0.0
    detected = (
        len(roi_samples) >= config.min_roi_samples
        and longest >= config.min_consecutive_samples
        and angle_range >= config.min_elbow_angle_degrees
        and path_length >= config.min_wrist_path_length
    )
    return HandleRuleResult(detected, len(values), len(roi_samples), longest, angle_range, path_length)


def analyze_handle_motion(
    video_path: Path, pose_model_path: Path, frame_interval: int, config: HandleRuleConfig,
) -> HandleRuleResult:
    """Optional landmark pass; called only for an enabled A17/A18 Trial 6 result."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("Unable to open uploaded video for handle rule.")
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        capture.release()
        raise ValueError("Invalid video FPS for handle rule.")
    samples: list[WristSample] = []
    options = vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(pose_model_path)),
        running_mode=vision.RunningMode.VIDEO, num_poses=2,
        min_pose_detection_confidence=0.5, min_tracking_confidence=0.5,
    )
    frame = -1
    try:
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                frame += 1
                if frame % frame_interval:
                    continue
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
                    int(frame / fps * 1000),
                )
                if not result.pose_landmarks:
                    continue
                # Matches diagnose_a18_handle_roi.py: first pose; prefer an in-ROI visible wrist.
                landmarks = result.pose_landmarks[0]
                candidates: list[tuple[float, float, float]] = []
                for wrist_i, elbow_i, shoulder_i in ((15, 13, 11), (16, 14, 12)):
                    wrist = landmarks[wrist_i]
                    if float(getattr(wrist, "visibility", 0.0)) >= 0.5:
                        candidates.append((wrist.x, wrist.y, _angle(landmarks[shoulder_i], landmarks[elbow_i], wrist)))
                if candidates:
                    x, y, angle = min(candidates, key=lambda item: 0 if config.roi.contains(item[0], item[1]) else 1)
                    samples.append(WristSample(frame, x, y, angle, config.roi.contains(x, y)))
    finally:
        capture.release()
    return evaluate_samples(samples, config)


def apply_a18_override(
    prediction: str, event: str, probabilities: dict[str, float], result: HandleRuleResult,
) -> tuple[str, str, dict[str, float], bool]:
    """Keep model score consistency by exchanging A17/A18 only on a positive rule."""
    if not result.detected or prediction != "A17":
        return prediction, event, probabilities, False
    adjusted = dict(probabilities)
    adjusted["A17"], adjusted["A18"] = adjusted["A18"], adjusted["A17"]
    return "A18", "door_lock_suspicious_activity", adjusted, True
