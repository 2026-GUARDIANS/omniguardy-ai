"""Gesture rules tuned for an unmirrored, front-facing camera feed.

The normal rule counts the thumb when deciding whether a hand is a fist.
At a frontal angle the thumb's horizontal landmark relation is unstable, even
when the other four fingers are clearly folded.  This variant keeps the same
victory/open-palm definitions but deliberately ignores the thumb for fist
detection.
"""

from __future__ import annotations

from gesture_rules import get_finger_state


def classify_gesture_front(landmarks, handedness_label: str) -> str:
    """Classify front-camera hand landmarks without making the thumb a fist gate."""
    state = get_finger_state(landmarks, handedness_label)

    if state.index and state.middle and not state.ring and not state.pinky:
        return "victory"
    if not state.index and not state.middle and not state.ring and not state.pinky:
        return "fist"
    if state.index and state.middle and state.ring and state.pinky:
        return "open_palm"
    return "unknown"
