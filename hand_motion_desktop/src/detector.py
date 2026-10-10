"""10초 내외의 영상 파일을 받아 Silent Signal(비상 손동작) 여부를 판단한다.

흐름:
  영상 파일 -> 프레임별 MediaPipe HandLandmarker 실행
  -> 프레임별 제스처 분류(gesture_rules)
  -> fist를 일정 프레임 이상 유지한 뒤 open_palm으로 바뀌는 '전환 동작'이
     REQUIRED_REPEATS번 연속으로 반복되면 Silent Signal로 판정
     (정지 자세가 아니라 동작 기반, 반복 요구로 오탐을 한 번 더 줄임)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from gesture_rules import (
    RepeatedFistToOpenTracker,
    classify_gesture,
    frame_gesture_from_hands,
)
from model_loader import ensure_model

# 비상 신호 코드/이름 (문서의 visionEvents.details 형식과 맞춤)
SIGNAL_CODE = "S1"
SIGNAL_NAME = "emergency_hand_gesture"

# 오탐 방지: fist로 인정하기까지 최소 연속 프레임 수, open_palm까지 허용하는 최대 간격
MIN_FIST_FRAMES = 5
MAX_TRANSITION_GAP_FRAMES = 15

# 몇 번 연속으로 주먹 쥐었다 펴야 확정할지, 반복 사이 최대 허용 간격(프레임)
# Demo policy: one valid fist -> open_palm transition is sufficient.
REQUIRED_REPEATS = 1
MAX_GAP_BETWEEN_REPS_FRAMES = 45


@dataclass
class SilentSignalResult:
    detected: bool
    confidence: float = 0.0
    detected_frame: int | None = None
    signal_code: str | None = None
    signal_name: str | None = None
    frames_analyzed: int = 0
    hand_present_frames: int = 0


def _build_detector() -> mp_vision.HandLandmarker:
    model_path = ensure_model()
    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def analyze_video_for_silent_signal(video_path: str | Path) -> SilentSignalResult:
    """영상 전체를 프레임 단위로 순회하며 Silent Signal 여부를 판단한다."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval_ms = 1000.0 / fps

    detector = _build_detector()
    tracker = RepeatedFistToOpenTracker(
        required_repeats=REQUIRED_REPEATS,
        min_fist_frames=MIN_FIST_FRAMES,
        max_transition_gap_frames=MAX_TRANSITION_GAP_FRAMES,
        max_gap_between_reps_frames=MAX_GAP_BETWEEN_REPS_FRAMES,
    )

    frame_idx = 0
    hand_present_frames = 0
    best_fist_streak = 0
    current_fist_streak = 0
    detected_at_frame: int | None = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int(frame_idx * frame_interval_ms)

            result = detector.detect_for_video(mp_image, timestamp_ms)

            gestures_this_frame: list[str] = []
            if result.hand_landmarks:
                hand_present_frames += 1
                for landmarks, handedness in zip(
                    result.hand_landmarks, result.handedness
                ):
                    label = handedness[0].category_name
                    gestures_this_frame.append(classify_gesture(landmarks, label))

            frame_gesture = frame_gesture_from_hands(gestures_this_frame)
            if frame_gesture == "fist":
                current_fist_streak += 1
                best_fist_streak = max(best_fist_streak, current_fist_streak)
            else:
                current_fist_streak = 0

            triggered = tracker.update(frame_gesture)
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
