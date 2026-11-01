"""Front-facing camera variant of the Silent Signal detector.

MediaPipe HandLandmarker assigns handedness on the assumption that its input
is a mirrored (selfie-style) image. A normally stored video from a camera
facing a person is not mirrored, so feed a horizontally flipped frame to the
landmarker. This keeps the landmark x-axis and the Left/Right label aligned
with the existing gesture rules, especially the thumb rule.

The source video is never changed; only the in-memory frame passed to
MediaPipe is flipped.
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import cv2
import mediapipe as mp

from detector import (
    MAX_GAP_BETWEEN_REPS_FRAMES,
    MIN_FIST_FRAMES,
    REQUIRED_REPEATS,
    SIGNAL_CODE,
    SIGNAL_NAME,
    SilentSignalResult,
    _build_detector,
)
from gesture_rules import RepeatedFistToOpenTracker, frame_gesture_from_hands
from gesture_rules_front import classify_gesture_front
from same_hand_signal_tracker import HandObservation, SameHandFistToOpenTracker


# At 30 fps this permits roughly one second from the last valid fist frame to
# an open palm. It applies only to SILENT_SIGNAL_COORDINATE_MODE=front.
FRONT_MAX_TRANSITION_GAP_FRAMES = 30


@dataclass(frozen=True)
class FrontHandObservation:
    """One classified hand in original-video normalized coordinates."""

    handedness: str
    gesture: str
    center_x_original: float
    center_y_original: float


@dataclass(frozen=True)
class FrontHandTrace:
    fps: float
    frame_count: int
    hand_present_frames: int
    observations_by_frame: list[list[FrontHandObservation]]


def collect_front_hand_observations(video_path: str | Path) -> FrontHandTrace:
    """Run exactly one front MediaPipe pass and retain classified hand observations.

    This is used only by the opt-in operator-filter experiment.  It intentionally
    does not decide an event because the existing YOLO tracking result may still
    be running in another worker.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval_ms = 1000.0 / fps
    detector = _build_detector()
    frame_index = hand_present_frames = 0
    frames: list[list[FrontHandObservation]] = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            mirrored_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            result = detector.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=mirrored_rgb),
                int(frame_index * frame_interval_ms),
            )
            observations: list[FrontHandObservation] = []
            if result.hand_landmarks:
                hand_present_frames += 1
                for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                    label = handedness[0].category_name
                    gesture = classify_gesture_front(landmarks, label)
                    hand = HandObservation.from_landmarks(landmarks, label, gesture)
                    # The detector sees a mirrored image.  YOLO tracking and
                    # source-frame bboxes use original-video coordinates.
                    observations.append(FrontHandObservation(
                        handedness=label,
                        gesture=gesture,
                        center_x_original=1.0 - hand.center_x,
                        center_y_original=hand.center_y,
                    ))
            frames.append(observations)
            frame_index += 1
    finally:
        cap.release()
        detector.close()
    return FrontHandTrace(fps, frame_index, hand_present_frames, frames)


def _ownership(
    hand: FrontHandObservation,
    frame_boxes: list[dict[str, Any]],
    operator_track_id: int | None,
    frame_width: int,
    frame_height: int,
) -> str:
    """Classify ownership without assigning an uncertain hand to any person."""
    if operator_track_id is None:
        return "operator_unavailable"
    x, y = hand.center_x_original * frame_width, hand.center_y_original * frame_height
    matching_ids: list[int] = []
    for box in frame_boxes:
        width = max(float(box["x2"]) - float(box["x1"]), 1.0)
        height = max(float(box["y2"]) - float(box["y1"]), 1.0)
        margin_x, margin_y = max(12.0, width * 0.05), max(12.0, height * 0.05)
        if float(box["x1"]) - margin_x <= x <= float(box["x2"]) + margin_x and \
           float(box["y1"]) - margin_y <= y <= float(box["y2"]) + margin_y:
            matching_ids.append(int(box["track_id"]))
    if not matching_ids:
        return "unassigned"
    if operator_track_id in matching_ids and len(matching_ids) == 1:
        return "operator"
    if operator_track_id in matching_ids:
        return "ambiguous"
    return "nonoperator"


def evaluate_front_hand_trace(
    trace: FrontHandTrace,
    *,
    frame_boxes: dict[int, list[dict[str, Any]]],
    operator_track_id: int | None,
    frame_width: int,
    frame_height: int,
    same_hand_only: bool,
) -> tuple[SilentSignalResult, list[dict[str, Any]]]:
    """Evaluate operator-owned hands only; uncertain hands are excluded.

    The ownership trace is returned for A/B diagnostics.  It is not added to
    the API DTO.  No fallback to unfiltered gestures occurs on uncertainty.
    """
    legacy_tracker = RepeatedFistToOpenTracker(
        required_repeats=REQUIRED_REPEATS,
        min_fist_frames=MIN_FIST_FRAMES,
        max_transition_gap_frames=FRONT_MAX_TRANSITION_GAP_FRAMES,
        max_gap_between_reps_frames=MAX_GAP_BETWEEN_REPS_FRAMES,
    )
    hand_tracker = SameHandFistToOpenTracker(
        min_fist_frames=MIN_FIST_FRAMES,
        max_transition_gap_frames=FRONT_MAX_TRANSITION_GAP_FRAMES,
    ) if same_hand_only else None
    best_fist = current_fist = 0
    detected_frame: int | None = None
    ownership_rows: list[dict[str, Any]] = []
    for frame_index, observations in enumerate(trace.observations_by_frame):
        accepted: list[FrontHandObservation] = []
        for observation in observations:
            status = _ownership(
                observation, frame_boxes.get(frame_index, []), operator_track_id,
                frame_width, frame_height,
            )
            ownership_rows.append({
                "frame": frame_index,
                "handedness": observation.handedness,
                "gesture": observation.gesture,
                "center_x_original": observation.center_x_original,
                "center_y_original": observation.center_y_original,
                "ownership": status,
                "accepted": status == "operator",
            })
            if status == "operator":
                accepted.append(observation)
        if hand_tracker is None:
            frame_gesture = frame_gesture_from_hands(item.gesture for item in accepted)
            if frame_gesture == "fist":
                current_fist += 1
                best_fist = max(best_fist, current_fist)
            else:
                current_fist = 0
            triggered = legacy_tracker.update(frame_gesture)
        else:
            same_hand_observations = [HandObservation(
                center_x=item.center_x_original,
                center_y=item.center_y_original,
                handedness=item.handedness,
                gesture=item.gesture,
            ) for item in accepted]
            triggered, _ids = hand_tracker.update(same_hand_observations)
            best_fist = max(
                best_fist,
                max((track.max_fist_streak for track in hand_tracker.tracks.values()), default=0),
            )
        if triggered and detected_frame is None:
            detected_frame = frame_index
    detected = detected_frame is not None
    return SilentSignalResult(
        detected=detected,
        confidence=round(min(1.0, best_fist / MIN_FIST_FRAMES), 2) if detected else 0.0,
        detected_frame=detected_frame,
        signal_code=SIGNAL_CODE if detected else None,
        signal_name=SIGNAL_NAME if detected else None,
        frames_analyzed=trace.frame_count,
        hand_present_frames=trace.hand_present_frames,
    ), ownership_rows


def analyze_video_for_silent_signal_front(video_path: str | Path) -> SilentSignalResult:
    """Detect a signal from an unmirrored, front-facing-camera video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval_ms = 1000.0 / fps
    detector = _build_detector()
    tracker = RepeatedFistToOpenTracker(
        required_repeats=REQUIRED_REPEATS,
        min_fist_frames=MIN_FIST_FRAMES,
        max_transition_gap_frames=FRONT_MAX_TRANSITION_GAP_FRAMES,
        max_gap_between_reps_frames=MAX_GAP_BETWEEN_REPS_FRAMES,
    )
    # Preserve the established frame-level aggregation unless explicitly
    # enabled for A/B validation.  The experimental branch never uses a
    # MediaPipe list index as a persistent hand identity.
    same_hand_only = os.getenv("VISION_SILENT_SIGNAL_SAME_HAND_ONLY", "false").strip().lower() in {
        "1", "true", "yes", "on",
    }
    same_hand_tracker = (
        SameHandFistToOpenTracker(
            min_fist_frames=MIN_FIST_FRAMES,
            max_transition_gap_frames=FRONT_MAX_TRANSITION_GAP_FRAMES,
        )
        if same_hand_only else None
    )

    frame_idx = hand_present_frames = best_fist_streak = current_fist_streak = 0
    detected_at_frame: int | None = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Convert the normal camera view to the selfie coordinate convention
            # expected by HandLandmarker; do not alter the original file/frame.
            mirrored_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=mirrored_rgb)
            result = detector.detect_for_video(mp_image, int(frame_idx * frame_interval_ms))

            gestures_this_frame: list[str] = []
            observations: list[HandObservation] = []
            if result.hand_landmarks:
                hand_present_frames += 1
                for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                    handedness_label = handedness[0].category_name
                    gesture = classify_gesture_front(landmarks, handedness_label)
                    gestures_this_frame.append(gesture)
                    if same_hand_tracker is not None:
                        observations.append(HandObservation.from_landmarks(landmarks, handedness_label, gesture))

            if same_hand_tracker is None:
                frame_gesture = frame_gesture_from_hands(gestures_this_frame)
                if frame_gesture == "fist":
                    current_fist_streak += 1
                    best_fist_streak = max(best_fist_streak, current_fist_streak)
                else:
                    current_fist_streak = 0
                triggered = tracker.update(frame_gesture)
            else:
                triggered, _hand_track_ids = same_hand_tracker.update(observations)
                best_fist_streak = max(
                    best_fist_streak,
                    max((hand_track.max_fist_streak for hand_track in same_hand_tracker.tracks.values()), default=0),
                )

            if triggered and detected_at_frame is None:
                detected_at_frame = frame_idx
            frame_idx += 1
    finally:
        cap.release()
        detector.close()

    detected = detected_at_frame is not None
    confidence = min(1.0, best_fist_streak / MIN_FIST_FRAMES) if detected else 0.0
    return SilentSignalResult(
        detected=detected,
        confidence=round(confidence, 2),
        detected_frame=detected_at_frame,
        signal_code=SIGNAL_CODE if detected else None,
        signal_name=SIGNAL_NAME if detected else None,
        frames_analyzed=frame_idx,
        hand_present_frames=hand_present_frames,
    )
