"""Unit regressions for opt-in AUDIO exhibition optimizations.

These tests intentionally mock video/model work.  They verify that the
legacy/default call remains unbounded and sequential, while the two switches
only affect the AUDIO helper's execution strategy, not its event payload.
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from api import vision_api as vision


VIDEO = {"durationSeconds": 13.0, "fps": 30.0}


class FakeService:
    def __init__(self) -> None:
        self.limits: list[int | None] = []

    def _tracking_features(self, _path: Path, *, max_frames: int | None = None) -> pd.DataFrame:
        self.limits.append(max_frames)
        return pd.DataFrame([
            {"frame": 0, "track_id": 7},
            {"frame": 149, "track_id": 7},
            {"frame": 300, "track_id": 9},
        ])


class TestExhibitionAudioOptimizations(unittest.TestCase):
    def setUp(self) -> None:
        self.previous = {
            name: os.environ.get(name)
            for name in (
                "VISION_EXHIBITION_LIMIT_TRACKING_TO_WINDOW",
                "VISION_EXHIBITION_PARALLEL_ANALYSIS",
                "VISION_KEYPAD_PARALLEL_ANALYSIS",
            )
        }
        for name in self.previous:
            os.environ.pop(name, None)

    def tearDown(self) -> None:
        for name, value in self.previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    @staticmethod
    def occlusion(_path: Path, _fps: float):
        return ([{"eventType": "CAMERA_OCCLUSION_SUSPECTED", "detectedFrame": 4}], {"analysisSeconds": 0.01})

    def call(self, service: FakeService):
        with patch.object(vision, "detect_camera_occlusion", self.occlusion), patch.object(
            vision, "detect_person_dwell", return_value=[{"eventType": "PERSON_DWELL_OBSERVED", "detectedFrame": 149}],
        ):
            return vision._analyze_exhibition_audio(service, Path("video.mp4"), VIDEO)

    def test_default_is_legacy_unbounded_and_sequential(self):
        service = FakeService()
        events, tracking, timing = self.call(service)
        self.assertEqual(service.limits, [None])
        self.assertFalse(timing["parallelAnalysis"])
        self.assertEqual(timing["trackingFrameLimit"], 0)
        self.assertEqual(events, [
            {"eventType": "CAMERA_OCCLUSION_SUSPECTED", "detectedFrame": 4},
            {"eventType": "PERSON_DWELL_OBSERVED", "detectedFrame": 149},
        ])
        self.assertEqual(list(tracking.frame), [0, 149])

    def test_window_and_parallel_are_independent_opt_ins(self):
        os.environ["VISION_EXHIBITION_LIMIT_TRACKING_TO_WINDOW"] = "true"
        os.environ["VISION_EXHIBITION_PARALLEL_ANALYSIS"] = "true"
        service = FakeService()
        events, tracking, timing = self.call(service)
        self.assertEqual(service.limits, [300])
        self.assertTrue(timing["parallelAnalysis"])
        self.assertEqual(timing["trackingFrameLimit"], 300)
        self.assertEqual(events, [
            {"eventType": "CAMERA_OCCLUSION_SUSPECTED", "detectedFrame": 4},
            {"eventType": "PERSON_DWELL_OBSERVED", "detectedFrame": 149},
        ])
        self.assertEqual(list(tracking.frame), [0, 149])

    def test_bounded_source_is_closed_when_yolo_iteration_raises(self):
        """The Windows temporary upload must not remain locked on a model error."""
        created = []

        class FakeSource:
            def __init__(self, _path, *, max_frames):
                self.max_frames = max_frames
                self.closed = False
                created.append(self)

            def close(self):
                self.closed = True

        class FailingYolo:
            @staticmethod
            def track(**_kwargs):
                def results():
                    raise RuntimeError("synthetic tracker failure")
                    yield None  # pragma: no cover - marks this as a generator
                return results()

        service = object.__new__(vision.InferenceService)
        service.yolo = FailingYolo()
        service.yolo_vid_stride = 1
        service.yolo_imgsz = 640
        with patch.object(vision, "LimitedVideoSource", FakeSource), patch.object(
            vision.torch.cuda, "is_available", return_value=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "synthetic tracker failure"):
                service._tracking_features(Path("video.mp4"), max_frames=300)
        self.assertEqual(len(created), 1)
        self.assertTrue(created[0].closed)

    def test_keypad_parallel_is_an_opt_in_and_preserves_inputs(self):
        class KeypadService:
            yolo = object()
            yolo_imgsz = 640

            def __init__(self):
                self.calls = 0

            def _tracking_features(self, _path):
                self.calls += 1
                return pd.DataFrame([{
                    "frame": 0, "track_id": 3,
                    "x1": 0.0, "y1": 0.0, "x2": 10.0, "y2": 10.0,
                }])

        signal = SimpleNamespace(hand_present_frames=1)
        service = KeypadService()
        with patch.object(vision, "analyze_silent_signal", return_value=signal), patch.object(
            vision, "_frame_dimensions", return_value=(1920, 1080),
        ), patch.object(vision, "analyze_rear_approach", return_value=None):
            returned_signal, tracking, rear, weapon, timing = vision._analyze_keypad(
                service, Path("video.mp4"), VIDEO,
            )
        self.assertIs(returned_signal, signal)
        self.assertEqual(service.calls, 1)
        self.assertIsNone(rear)
        self.assertIsNone(weapon)
        self.assertFalse(timing["parallelAnalysis"])
        self.assertEqual(tracking.attrs["frame_width"], 1920)
        self.assertEqual(tracking.attrs["frame_height"], 1080)

        os.environ["VISION_KEYPAD_PARALLEL_ANALYSIS"] = "true"
        parallel_service = KeypadService()
        with patch.object(vision, "analyze_silent_signal", return_value=signal), patch.object(
            vision, "_frame_dimensions", return_value=(1920, 1080),
        ), patch.object(vision, "analyze_rear_approach", return_value=None):
            _, _, _, _, parallel_timing = vision._analyze_keypad(
                parallel_service, Path("video.mp4"), VIDEO,
            )
        self.assertEqual(parallel_service.calls, 1)
        self.assertTrue(parallel_timing["parallelAnalysis"])


if __name__ == "__main__":
    unittest.main()
