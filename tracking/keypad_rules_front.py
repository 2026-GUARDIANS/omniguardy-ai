"""Front-camera operator selection for the rear-close-approach rule.

With a frontal camera there is no reliable semantic "rear" axis in 2-D
tracking.  Instead, lock the foreground person (largest person bounding box)
as the keypad operator at the beginning of the clip.  A later entrant cannot
replace that operator selection.
"""

from __future__ import annotations

import pandas as pd

from tracking.keypad_rules import detect_rear_approach


FRONT_OPERATOR_LOCK_WINDOW_SECONDS = 0.3


def identify_front_operator(tracking: pd.DataFrame, fps: float) -> int | None:
    """Return the largest initially visible person track, fixed for this clip."""
    required = {"frame", "track_id", "x1", "y1", "x2", "y2"}
    if tracking.empty or fps <= 0 or not required.issubset(tracking.columns):
        return None

    first_frame = int(tracking["frame"].min())
    lock_end_frame = first_frame + max(1, round(FRONT_OPERATOR_LOCK_WINDOW_SECONDS * fps))
    opening = tracking[tracking["frame"] <= lock_end_frame]
    if opening.empty:
        return None

    candidates: list[tuple[float, int, int]] = []
    for track_id, group in opening.groupby("track_id"):
        # Tracks that only enter after the opening window must not take over.
        if int(group["frame"].min()) > lock_end_frame:
            continue
        area = ((group["x2"] - group["x1"]).clip(lower=0)
                * (group["y2"] - group["y1"]).clip(lower=0))
        if not area.empty:
            candidates.append((float(area.median()), -int(group["frame"].min()), int(track_id)))

    if not candidates:
        return None
    # Largest foreground bbox wins. Earlier appearance then lower ID break ties.
    return max(candidates, key=lambda value: (value[0], value[1], -value[2]))[2]


def detect_front_rear_approach(tracking: pd.DataFrame, fps: float):
    """Run the existing proximity rule with the foreground operator locked."""
    operator_id = identify_front_operator(tracking, fps)
    if operator_id is None:
        return None
    event = detect_rear_approach(tracking, fps, operator_id=operator_id)
    if event is not None:
        event["details"]["operatorSelection"] = "front_foreground_bbox"
    return event
