"""Fast unit checks for the new KEYPAD early-return branch (no models/video)."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from io import BytesIO
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from api import vision_api as vision


class Upload:
    filename = "sample.mp4"
    file = BytesIO(b"not-a-real-video")
    async def close(self): pass


VIDEO = {"fileName": "sample.mp4", "durationSeconds": 13.0, "fps": 30.0, "frameCount": 390}


class TestEarlyReturnContract(unittest.TestCase):
    def setUp(self):
        self.old = {name: os.environ.get(name) for name in (
            "VISION_KEYPAD_EARLY_RETURN_ON_SILENT_SIGNAL", "SILENT_SIGNAL_COORDINATE_MODE",
            "VISION_SILENT_SIGNAL_SAME_HAND_ONLY", "VISION_SILENT_SIGNAL_OPERATOR_ONLY",
        )}
        os.environ.update({"VISION_KEYPAD_EARLY_RETURN_ON_SILENT_SIGNAL": "true",
                           "SILENT_SIGNAL_COORDINATE_MODE": "front",
                           "VISION_SILENT_SIGNAL_SAME_HAND_ONLY": "false",
                           "VISION_SILENT_SIGNAL_OPERATOR_ONLY": "false"})
        vision.app.state.service = None

    def tearDown(self):
        for name, value in self.old.items():
            if value is None: os.environ.pop(name, None)
            else: os.environ[name] = value
        vision.app.state.service = None

    def call(self):
        return asyncio.run(vision.analyze_vision(
            file=Upload(), triggerId="unit", triggerType="KEYPAD", securityEventId=None,
            triggeredAt=datetime.now(timezone.utc),
        ))

    def test_signal_returns_without_initializing_yolo(self):
        signal = SimpleNamespace(detected=True, confidence=1.0, detected_frame=114,
                                 signal_code="S1", signal_name="emergency_hand_gesture",
                                 hand_present_frames=8)
        with patch.object(vision, "_video_metadata", return_value=VIDEO), patch.object(
            vision, "analyze_until_silent_signal", return_value=signal,
        ), patch.object(vision, "InferenceService") as service:
            payload = self.call()
        service.assert_not_called()
        result = payload["result"]
        self.assertEqual(payload["status"], "success")
        self.assertFalse(result["analysisComplete"])
        self.assertEqual(result["earlyReturnReason"], "SILENT_SIGNAL")
        self.assertIsNone(result["observations"]["trackedPersonCount"])
        self.assertEqual(result["visionEvents"][0]["detectedFrame"], 114)

    def test_no_signal_reuses_prefix_result_for_normal_completion(self):
        signal = SimpleNamespace(detected=False, hand_present_frames=2)
        tracking = pd.DataFrame(columns=["frame", "track_id"])
        returned = (signal, tracking, None, None, {"parallelAnalysis": False})
        fake_service = object()
        vision.app.state.service = fake_service
        with patch.object(vision, "_video_metadata", return_value=VIDEO), patch.object(
            vision, "analyze_until_silent_signal", return_value=signal,
        ), patch.object(vision, "_analyze_keypad", return_value=returned) as analysis:
            payload = self.call()
        self.assertNotIn("analysisComplete", payload["result"])
        self.assertEqual(payload["result"]["visionEvents"], [])
        self.assertIs(analysis.call_args.kwargs["precomputed_signal"], signal)


if __name__ == "__main__":
    unittest.main()
