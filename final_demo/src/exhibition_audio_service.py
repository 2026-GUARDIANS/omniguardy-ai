"""Runtime rules for the exhibition AUDIO path.

The functions analyze the uploaded MP4 itself; they never read offline CSV or
JSON results.  They intentionally do not contain any risk-level policy.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import time
from typing import Any

import cv2
import pandas as pd


ANALYSIS_SECONDS = 10.0
EDGE_DENSITY_MAX = 0.001
LAPLACIAN_VARIANCE_MAX = 5.0
OCCLUSION_MIN_SUSTAINED_SECONDS = 0.3
DWELL_MIN_CUMULATIVE_SECONDS = 5.0


def _frame_limit(fps: float) -> int:
    """Use the same first-10-second frame bound as offline validation."""
    return max(1, round(ANALYSIS_SECONDS * fps))


def detect_camera_occlusion(video_path: Path, fps: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return every sustained low-structure interval in the first 10 seconds.

    The Canny thresholds, full-frame edge density, Laplacian variance, frame
    bound, and duration rounding intentionally match
    ``validate_edge_occlusion_candidate.py``. Brightness and grid-change
    features are deliberately not used.
    """
    started = time.perf_counter()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video for occlusion analysis: {video_path}")

    flags: list[bool] = []
    metrics: list[tuple[float, float]] = []
    limit = _frame_limit(fps)
    frame = 0
    try:
        while frame < limit:
            ok, image = capture.read()
            if not ok:
                break
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edge_density = float(cv2.countNonZero(cv2.Canny(gray, 100, 200))) / float(gray.size)
            laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            metrics.append((edge_density, laplacian_variance))
            flags.append(edge_density < EDGE_DENSITY_MAX and laplacian_variance < LAPLACIAN_VARIANCE_MAX)
            frame += 1
    finally:
        capture.release()

    required_frames = max(1, round(OCCLUSION_MIN_SUSTAINED_SECONDS * fps))
    intervals: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate([*flags, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start >= required_frames:
                intervals.append((start, index - 1))
            start = None

    events: list[dict[str, Any]] = []
    window_seconds = round(len(metrics) / fps, 3) if fps > 0 else 0.0
    for start_frame, end_frame in intervals:
        confirmation_frame = start_frame + required_frames - 1
        events.append({
            "eventType": "CAMERA_OCCLUSION_SUSPECTED",
            "confidence": None,
            # Detection becomes certain only at this frame, after the required
            # continuous duration has elapsed.
            "detectedFrame": confirmation_frame,
            "details": {
                "startSeconds": round(start_frame / fps, 3),
                "endSeconds": round(end_frame / fps, 3),
                "durationSeconds": round((end_frame - start_frame + 1) / fps, 3),
                "edgeDensityThreshold": EDGE_DENSITY_MAX,
                "laplacianThreshold": LAPLACIAN_VARIANCE_MAX,
                "minSustainedSeconds": OCCLUSION_MIN_SUSTAINED_SECONDS,
                "windowSeconds": window_seconds,
                "experimental": True,
            },
        })
    return events, {
        "framesAnalyzed": len(metrics),
        "analysisSeconds": round(time.perf_counter() - started, 3),
    }


def detect_person_dwell(
    tracking: pd.DataFrame, fps: float, *, window_seconds: float = ANALYSIS_SECONDS,
) -> list[dict[str, Any]]:
    """Emit one observation per track ID after 5 seconds of visible frames.

    Different IDs are never joined.  A same-ID reappearance is naturally
    included because its unique observed frames accumulate within this video.
    """
    if tracking.empty or fps <= 0:
        return []
    required = {"frame", "track_id"}
    if not required.issubset(tracking.columns):
        raise ValueError("Tracking output lacks frame or track_id for dwell analysis.")

    effective_window_seconds = min(ANALYSIS_SECONDS, max(0.0, window_seconds))
    cutoff = max(1, round(effective_window_seconds * fps))
    frames_by_id: dict[int, list[int]] = defaultdict(list)
    subset = tracking[tracking["frame"] < cutoff]
    for row in subset[["frame", "track_id"]].itertuples(index=False):
        frames_by_id[int(row.track_id)].append(int(row.frame))

    events: list[dict[str, Any]] = []
    for track_id in sorted(frames_by_id):
        frames = sorted(set(frames_by_id[track_id]))
        if not frames:
            continue
        confirmation_index = next((i for i in range(len(frames)) if (i + 1) / fps >= DWELL_MIN_CUMULATIVE_SECONDS), None)
        if confirmation_index is None:
            continue
        gaps = [max(0, later - earlier - 1) for earlier, later in zip(frames, frames[1:])]
        longest_run = current_run = 1
        for earlier, later in zip(frames, frames[1:]):
            if later == earlier + 1:
                current_run += 1
                longest_run = max(longest_run, current_run)
            else:
                current_run = 1
        detected_frame = frames[confirmation_index]
        events.append({
            "eventType": "PERSON_DWELL_OBSERVED",
            "confidence": None,
            "detectedFrame": detected_frame,
            "details": {
                "trackId": track_id,
                "thresholdSeconds": DWELL_MIN_CUMULATIVE_SECONDS,
                "cumulativeObservedSeconds": round(len(frames) / fps, 3),
                "continuousObservedSeconds": round(longest_run / fps, 3),
                "firstLastSpanSeconds": round((frames[-1] - frames[0]) / fps, 3),
                "maximumDetectionGapSeconds": round(max(gaps, default=0) / fps, 3),
                "identityContinuity": "same_track_only",
                "windowSeconds": round(effective_window_seconds, 3),
                "experimental": True,
            },
        })
    return events
