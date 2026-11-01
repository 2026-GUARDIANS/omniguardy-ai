"""Experimental per-hand association for front Silent Signal detection.

MediaPipe's hand-list index is frame-local.  This module therefore associates
observations by palm position and age, with handedness only as a soft matching
signal.  It intentionally does not use YOLO/operator information.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

from gesture_rules import FistToOpenTracker


WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 0, 5, 9, 17
MAX_MISSING_FRAMES = 5
MAX_ASSOCIATION_DISTANCE = 0.12
HANDEDNESS_MISMATCH_COST = 0.03


@dataclass(frozen=True)
class HandObservation:
    """One MediaPipe hand result in normalized, mirrored-input coordinates."""

    center_x: float
    center_y: float
    handedness: str
    gesture: str

    @classmethod
    def from_landmarks(cls, landmarks, handedness: str, gesture: str) -> "HandObservation":
        # A small palm anchor is less jittery than a fingertip and remains
        # meaningful for both fist and open-palm configurations.
        anchors = [landmarks[index] for index in (WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP)]
        return cls(
            center_x=sum(point.x for point in anchors) / len(anchors),
            center_y=sum(point.y for point in anchors) / len(anchors),
            handedness=handedness,
            gesture=gesture,
        )


@dataclass
class HandTrack:
    track_id: int
    center_x: float
    center_y: float
    handedness: str
    transition: FistToOpenTracker
    missed_frames: int = 0
    fist_streak: int = 0
    max_fist_streak: int = 0


class SameHandFistToOpenTracker:
    """Associate hands and accept a transition only within one hand track.

    A track survives up to ``max_missing_frames`` absent detector frames.  On
    the next absent frame it expires, discarding any armed fist state; a later
    reappearance starts a fresh track.  A matched ``unknown`` gesture still
    advances the inherited transition's allowed gap, matching the legacy
    FistToOpenTracker semantics.
    """

    def __init__(
        self,
        *,
        min_fist_frames: int,
        max_transition_gap_frames: int,
        max_missing_frames: int = MAX_MISSING_FRAMES,
        max_association_distance: float = MAX_ASSOCIATION_DISTANCE,
    ) -> None:
        self.min_fist_frames = min_fist_frames
        self.max_transition_gap_frames = max_transition_gap_frames
        self.max_missing_frames = max_missing_frames
        self.max_association_distance = max_association_distance
        self._tracks: dict[int, HandTrack] = {}
        self._next_track_id = 1

    @property
    def tracks(self) -> dict[int, HandTrack]:
        """Read-only-by-convention state for diagnostics/tests."""
        return self._tracks

    def _new_track(self, observation: HandObservation) -> HandTrack:
        track = HandTrack(
            track_id=self._next_track_id,
            center_x=observation.center_x,
            center_y=observation.center_y,
            handedness=observation.handedness,
            transition=FistToOpenTracker(
                min_fist_frames=self.min_fist_frames,
                max_transition_gap_frames=self.max_transition_gap_frames,
                cooldown_frames=0,
            ),
        )
        self._next_track_id += 1
        self._tracks[track.track_id] = track
        return track

    @staticmethod
    def _distance(track: HandTrack, observation: HandObservation) -> float:
        return math.hypot(track.center_x - observation.center_x, track.center_y - observation.center_y)

    def update(self, observations: list[HandObservation]) -> tuple[bool, list[int | None]]:
        """Update all tracks and return (signal_detected, IDs per observation)."""
        candidates: list[tuple[float, int, int]] = []
        for observation_index, observation in enumerate(observations):
            for track_id, track in self._tracks.items():
                # Do not grow the matching radius during a detection loss.
                # An armed fist track must not steal a different nearby hand
                # when the original hand is briefly occluded.
                distance = self._distance(track, observation)
                if distance <= self.max_association_distance:
                    handedness_cost = 0.0 if track.handedness == observation.handedness else HANDEDNESS_MISMATCH_COST
                    candidates.append((distance + handedness_cost, track_id, observation_index))
        candidates.sort()
        assigned_tracks: set[int] = set()
        assigned_observations: set[int] = set()
        mapping: list[int | None] = [None] * len(observations)
        for _cost, track_id, observation_index in candidates:
            if track_id in assigned_tracks or observation_index in assigned_observations:
                continue
            assigned_tracks.add(track_id)
            assigned_observations.add(observation_index)
            mapping[observation_index] = track_id

        for observation_index, observation in enumerate(observations):
            if mapping[observation_index] is None:
                mapping[observation_index] = self._new_track(observation).track_id

        detected = False
        matched = set(mapping)
        for observation, track_id in zip(observations, mapping):
            track = self._tracks[int(track_id)]
            track.center_x, track.center_y = observation.center_x, observation.center_y
            track.handedness = observation.handedness
            track.missed_frames = 0
            if observation.gesture == "fist":
                track.fist_streak += 1
                track.max_fist_streak = max(track.max_fist_streak, track.fist_streak)
            else:
                track.fist_streak = 0
            detected = track.transition.update(observation.gesture) or detected

        expired = []
        for track_id, track in self._tracks.items():
            if track_id not in matched:
                track.missed_frames += 1
                # Advance an armed transition through an actual absent frame,
                # then expire it if the configurable gap is exceeded.
                track.transition.update("none")
                if track.missed_frames > self.max_missing_frames:
                    expired.append(track_id)
        for track_id in expired:
            del self._tracks[track_id]
        return detected, mapping
