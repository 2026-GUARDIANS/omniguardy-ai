"""Prefix-only replay of the existing Silent Signal rule.

This module deliberately imports the production detector's model builder,
constants, classifier, and state machine.  It changes only the stopping point:
once the existing rule confirms its first signal it closes MediaPipe/OpenCV and
returns, rather than reading later frames.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp

from detector import (
    MAX_GAP_BETWEEN_REPS_FRAMES, MAX_TRANSITION_GAP_FRAMES, MIN_FIST_FRAMES,
    REQUIRED_REPEATS, SIGNAL_CODE, SIGNAL_NAME, SilentSignalResult,
    _build_detector,
)
from detector_front import FRONT_MAX_TRANSITION_GAP_FRAMES
from gesture_rules import RepeatedFistToOpenTracker, classify_gesture, frame_gesture_from_hands
from gesture_rules_front import classify_gesture_front


def analyze_until_silent_signal(video_path: str | Path, *, coordinate_mode: str) -> SilentSignalResult:
    """Return at the current rule's first confirmed event, or at EOF.

    ``coordinate_mode`` has the same legacy/front meaning as ``vision_api``.
    The caller must fall back to the normal path when a feature (for example
    same-hand or operator ownership filtering) needs tracking data.
    """
    front = coordinate_mode == "front"
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    detector = _build_detector()
    tracker = RepeatedFistToOpenTracker(
        required_repeats=REQUIRED_REPEATS,
        min_fist_frames=MIN_FIST_FRAMES,
        max_transition_gap_frames=FRONT_MAX_TRANSITION_GAP_FRAMES if front else MAX_TRANSITION_GAP_FRAMES,
        max_gap_between_reps_frames=MAX_GAP_BETWEEN_REPS_FRAMES,
    )
    frame_index = hand_present_frames = best_fist_streak = current_fist_streak = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(cv2.flip(frame, 1) if front else frame, cv2.COLOR_BGR2RGB)
            result = detector.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                int(frame_index * 1000.0 / fps),
            )
            gestures: list[str] = []
            if result.hand_landmarks:
                hand_present_frames += 1
                classifier = classify_gesture_front if front else classify_gesture
                for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                    gestures.append(classifier(landmarks, handedness[0].category_name))
            frame_gesture = frame_gesture_from_hands(gestures)
            if frame_gesture == "fist":
                current_fist_streak += 1
                best_fist_streak = max(best_fist_streak, current_fist_streak)
            else:
                current_fist_streak = 0
            if tracker.update(frame_gesture):
                return SilentSignalResult(
                    detected=True, confidence=round(min(1.0, best_fist_streak / MIN_FIST_FRAMES), 2),
                    detected_frame=frame_index, signal_code=SIGNAL_CODE, signal_name=SIGNAL_NAME,
                    frames_analyzed=frame_index + 1, hand_present_frames=hand_present_frames,
                )
            frame_index += 1
    finally:
        capture.release()
        detector.close()
    return SilentSignalResult(
        detected=False, confidence=0.0, frames_analyzed=frame_index,
        hand_present_frames=hand_present_frames,
    )
