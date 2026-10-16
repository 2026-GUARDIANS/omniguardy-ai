"""KEYPAD 트리거 전용 규칙 기반 visionEvents 판정.

학습된 모델이 아니라 YOLO person tracking 결과(또는 knife/scissors 탐지 결과)에
직접 거리/시간 규칙을 적용하는 순수 로직이라, api/vision_api.py의 InferenceService
(모델 로딩 담당)와 분리해서 이 파일에 모아둔다.

포함된 두 가지:
  - detect_rear_approach(): REAR_CLOSE_APPROACH_SUSPECTED (팀 확정 스키마)
  - detect_weapon_proximity(): WEAPON_PROXIMITY_SUSPECTED (실험적 기능, 교수님 요청.
    아직 팀 스키마에 없는 eventType이라 호출 여부는 vision_api.py의
    ENABLE_WEAPON_DETECTION 플래그로 제어한다)
"""

from __future__ import annotations

import math
import os
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---- REAR_CLOSE_APPROACH_SUSPECTED 판정 파라미터 ----
# 사용자 재식별이 없으므로 "가장 먼저 나타난 track = 키패드 조작자"로 가정하는
# 휴리스틱을 쓴다. 이후 새로 나타난 track이 조작자와의 거리를 좁히며 일정 시간
# 이상 가깝게 머무르면 후방 접근으로 판단한다.
REAR_APPROACH_MIN_DELAY_SECONDS = 0.5   # 조작자 등장 후 최소 이 시간 뒤에 나타나야 "나중에 등장"으로 인정
REAR_APPROACH_CLOSE_RATIO = 1.3         # (두 중심 거리) < 조작자 bbox 높이 * 이 배율이면 "근접"
REAR_APPROACH_MIN_CLOSE_SECONDS = 1.0   # 근접 상태가 최소 이 시간 이상 유지되어야 확정
REAR_APPROACH_CLOSING_MARGIN = 0.05     # 초반 대비 후반 정규화 거리가 이 비율 이상 줄어야 "좁혀짐"

# ---- 흉기 근접 감지 (실험적 기능, 교수님 요청) ----
# COCO 사전학습 YOLOv8n에는 별도 학습 없이 knife(43)/scissors(76) 클래스가 이미
# 포함돼 있다. 다만 COCO의 knife 학습 이미지는 대부분 주방/식탁 장면이라, 사람이
# 들고 들이대는 각도의 칼 인식은 보장되지 않는다 — 데모용 실험 기능으로만 취급.
ENABLE_WEAPON_DETECTION = os.getenv("VISION_ENABLE_WEAPON_DETECTION", "false").lower() == "true"
WEAPON_CLASS_IDS = [43, 76]   # knife, scissors
WEAPON_CONF_THRESHOLD = 0.25
WEAPON_PROXIMITY_RATIO = 1.5  # (흉기 중심과 가장 가까운 사람 중심 거리) < 사람 bbox 높이 * 이 배율


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KEYPAD_ROI_PATH = PROJECT_ROOT / "tracking" / "keypad_roi.json"


def load_keypad_roi(path: str | Path | None = None) -> dict[str, float] | None:
    """정규화된 키패드 사각형 ROI를 읽는다. 설정이 없거나 잘못되면 None이다."""
    roi_path = Path(path or os.getenv("VISION_KEYPAD_ROI_PATH", DEFAULT_KEYPAD_ROI_PATH))
    try:
        with roi_path.open(encoding="utf-8") as file:
            value = json.load(file)
        points = value.get("points", [])
        if len(points) != 2:
            return None
        x_values = [float(point["x"]) for point in points]
        y_values = [float(point["y"]) for point in points]
        if not all(0.0 <= coordinate <= 1.0 for coordinate in (*x_values, *y_values)):
            return None
        x1, x2 = sorted(x_values)
        y1, y2 = sorted(y_values)
        if x1 == x2 or y1 == y2:
            return None
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return None


def identify_operator_by_roi(
    tracking: pd.DataFrame, roi: dict[str, float], fps: float,
) -> int | None:
    """키패드 ROI와 겹치는 시간이 가장 이른 track을 조작자로 고른다.

    person bbox의 중앙~하단만 쓴다. 손 keypoint가 있으면 이 근사를 손 위치 판정으로
    대체할 수 있다.
    """
    required = {"frame", "track_id", "x1", "y1", "x2", "y2"}
    if tracking.empty or not required.issubset(tracking.columns) or fps <= 0:
        return None
    try:
        roi_x1, roi_y1 = float(roi["x1"]), float(roi["y1"])
        roi_x2, roi_y2 = float(roi["x2"]), float(roi["y2"])
    except (KeyError, TypeError, ValueError):
        return None
    # 원본 프레임 크기는 API/시각화 도구가 DataFrame attrs로 붙인다. bbox 최대값을
    # 해상도로 쓰면 화면 가장자리 사람이 없을 때 ROI가 틀어질 수 있다.
    frame_width = float(tracking.attrs.get("frame_width", 0))
    frame_height = float(tracking.attrs.get("frame_height", 0))
    if frame_width <= 0 or frame_height <= 0:
        return None
    roi_px = (roi_x1 * frame_width, roi_y1 * frame_height, roi_x2 * frame_width, roi_y2 * frame_height)

    candidates: list[tuple[int, int, float, int]] = []
    for track_id, group in tracking.groupby("track_id"):
        overlaps: list[tuple[int, float]] = []
        for row in group.itertuples():
            # bbox 중앙 60%, 하단 55%: 키패드로 뻗는 손 영역의 근사다.
            width, height = max(row.x2 - row.x1, 0.0), max(row.y2 - row.y1, 0.0)
            bx1, bx2 = row.x1 + width * 0.2, row.x2 - width * 0.2
            by1, by2 = row.y1 + height * 0.45, row.y2
            ix1, iy1 = max(bx1, roi_px[0]), max(by1, roi_px[1])
            ix2, iy2 = min(bx2, roi_px[2]), min(by2, roi_px[3])
            intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if intersection > 0:
                overlaps.append((int(row.frame), intersection))
        if overlaps:
            # 가장 이른 겹침이 우선, 동률이면 오래/크게 겹친 track을 고른다.
            candidates.append((min(frame for frame, _ in overlaps), -len(overlaps), -sum(area for _, area in overlaps), int(track_id)))
    return min(candidates)[3] if candidates else None


def detect_rear_approach(
    tracking: pd.DataFrame, fps: float, operator_id: int | None = None,
) -> dict[str, Any] | None:
    """규칙 기반 REAR_CLOSE_APPROACH_SUSPECTED 판정.

    tracking: InferenceService._tracking_features()가 반환하는 형태
              (columns: frame, track_id, x1, y1, x2, y2)
    사람이 한 명뿐이거나(track 1개), 조건을 만족하는 track이 없으면 None.
    """
    if tracking.empty or fps <= 0:
        return None

    tracking = tracking.sort_values("frame")
    tracks: dict[int, pd.DataFrame] = {
        track_id: group.reset_index(drop=True) for track_id, group in tracking.groupby("track_id")
    }
    if len(tracks) < 2:
        return None  # 사람이 한 명뿐이면 후방 접근 자체가 성립하지 않음

    first_seen = {track_id: int(group["frame"].iloc[0]) for track_id, group in tracks.items()}
    if operator_id not in tracks:
        roi = load_keypad_roi()
        # ROI 설정/겹침이 없으면 이전의 첫 등장 track 휴리스틱을 그대로 쓴다.
        operator_id = identify_operator_by_roi(tracking, roi, fps) if roi else None
        operator_id = operator_id if operator_id is not None else min(first_seen, key=first_seen.get)
    operator = tracks[operator_id]
    operator_first_frame = first_seen[operator_id]
    min_delay_frames = REAR_APPROACH_MIN_DELAY_SECONDS * fps

    best_event: dict[str, Any] | None = None
    for track_id, group in tracks.items():
        if track_id == operator_id:
            continue
        approacher_first_frame = first_seen[track_id]
        if approacher_first_frame - operator_first_frame < min_delay_frames:
            continue  # 조작자와 거의 동시에 등장 = 같이 들어온 것으로 간주, 후방 접근 아님

        merged = pd.merge(
            operator[["frame", "x1", "y1", "x2", "y2"]],
            group[["frame", "x1", "y1", "x2", "y2"]],
            on="frame", suffixes=("_op", "_ap"),
        )
        if merged.empty:
            continue

        op_center_x = (merged.x1_op + merged.x2_op) / 2
        op_center_y = (merged.y1_op + merged.y2_op) / 2
        ap_center_x = (merged.x1_ap + merged.x2_ap) / 2
        ap_center_y = (merged.y1_ap + merged.y2_ap) / 2
        op_height = (merged.y2_op - merged.y1_op).clip(lower=1.0)  # 원근에 따른 스케일 보정 기준

        distance = np.hypot(ap_center_x - op_center_x, ap_center_y - op_center_y)
        normalized_distance = distance / op_height

        close_mask = normalized_distance < REAR_APPROACH_CLOSE_RATIO
        close_seconds = float(close_mask.sum()) / fps
        if close_seconds < REAR_APPROACH_MIN_CLOSE_SECONDS:
            continue

        half = max(1, len(normalized_distance) // 2)
        early_avg = float(normalized_distance.iloc[:half].mean())
        late_avg = float(normalized_distance.iloc[half:].mean())
        distance_closing = (early_avg - late_avg) > REAR_APPROACH_CLOSING_MARGIN
        if not distance_closing:
            continue

        close_indices = merged.index[close_mask]
        detected_frame = int(merged.loc[close_indices[0], "frame"]) if len(close_indices) else int(merged.frame.iloc[-1])

        closing_strength = min(1.0, max(0.0, (early_avg - late_avg) / max(early_avg, 1e-6)))
        proximity_strength = min(1.0, close_seconds / (REAR_APPROACH_MIN_CLOSE_SECONDS * 2))
        confidence = round(min(0.95, 0.5 + 0.25 * closing_strength + 0.25 * proximity_strength), 2)

        candidate = {
            "eventType": "REAR_CLOSE_APPROACH_SUSPECTED",
            "confidence": confidence,
            "detectedFrame": detected_frame,
            "details": {
                "personCount": len(tracks),
                "delayedEntrySeconds": round((approacher_first_frame - operator_first_frame) / fps, 2),
                "distanceClosingDetected": bool(distance_closing),
                "closeProximityDurationSeconds": round(close_seconds, 2),
            },
        }
        if best_event is None or candidate["confidence"] > best_event["confidence"]:
            best_event = candidate

    return best_event


def detect_weapon_proximity(
    video_path: Path,
    yolo_model: Any,
    imgsz: int,
    tracking: pd.DataFrame,
    device: int | str,
) -> dict[str, Any] | None:
    """실험적 기능: COCO 사전학습 knife/scissors 클래스와 사람 bbox의 근접 여부를 본다.

    별도 학습 없이 기존 YOLOv8n(yolo_model)을 그대로 재사용한다. 호출 여부는
    vision_api.py에서 ENABLE_WEAPON_DETECTION 플래그로 결정한다.
    """
    results = yolo_model.predict(
        source=str(video_path), classes=WEAPON_CLASS_IDS, conf=WEAPON_CONF_THRESHOLD,
        stream=True, device=device, imgsz=imgsz, verbose=False,
    )

    person_by_frame: dict[int, list[tuple[float, float, float]]] = {}
    if not tracking.empty:
        for frame, group in tracking.groupby("frame"):
            person_by_frame[int(frame)] = [
                ((row.x1 + row.x2) / 2, (row.y1 + row.y2) / 2, max(row.y2 - row.y1, 1.0))
                for row in group.itertuples()
            ]

    best_frame: int | None = None
    best_confidence = 0.0
    for frame_idx, result in enumerate(results):
        if result.boxes is None or len(result.boxes) == 0:
            continue
        persons = person_by_frame.get(frame_idx, [])
        if not persons:
            continue  # 같은 프레임에 사람이 감지된 경우에만 "근접"을 판단
        for box in result.boxes:
            weapon_conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            weapon_center_x, weapon_center_y = (x1 + x2) / 2, (y1 + y2) / 2
            for person_center_x, person_center_y, person_height in persons:
                distance = math.hypot(weapon_center_x - person_center_x, weapon_center_y - person_center_y)
                if distance < person_height * WEAPON_PROXIMITY_RATIO and weapon_conf > best_confidence:
                    best_confidence = weapon_conf
                    best_frame = frame_idx

    if best_frame is None:
        return None
    return {
        "eventType": "WEAPON_PROXIMITY_SUSPECTED",
        "confidence": round(best_confidence, 2),
        "detectedFrame": best_frame,
        "details": {
            "experimental": True,
            "note": "COCO 사전학습 knife/scissors 클래스 기반, 별도 학습 없음 — 데모용 실험 기능",
        },
    }
