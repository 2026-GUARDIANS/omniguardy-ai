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

from pathlib import Path

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


# At 30 fps this permits roughly one second from the last valid fist frame to
# an open palm. It applies only to SILENT_SIGNAL_COORDINATE_MODE=front.
FRONT_MAX_TRANSITION_GAP_FRAMES = 30


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
            if result.hand_landmarks:
                hand_present_frames += 1
                for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                    gestures_this_frame.append(
                        classify_gesture_front(landmarks, handedness[0].category_name)
                    )

            frame_gesture = frame_gesture_from_hands(gestures_this_frame)
            if frame_gesture == "fist":
                current_fist_streak += 1
                best_fist_streak = max(best_fist_streak, current_fist_streak)
            else:
                current_fist_streak = 0

            if tracker.update(frame_gesture) and detected_at_frame is None:
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
