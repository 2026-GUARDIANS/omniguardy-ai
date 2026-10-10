"""Unit coverage for the AUDIO-only A18 handle ROI helper."""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from tracking.handle_roi_rules import (
    HandleRoi, HandleRuleConfig, HandleRuleResult, WristSample,
    apply_a18_override, evaluate_samples,
)


class TestHandleRoiRules(unittest.TestCase):
    def setUp(self) -> None:
        self.config = HandleRuleConfig(HandleRoi(.35, .44, .57, .68), 3, 2, 8.0, .05)
        self.positive = [
            WristSample(0, .40, .50, 40.0, True),
            WristSample(10, .43, .50, 49.0, True),
            WristSample(20, .47, .50, 52.0, True),
        ]

    def test_all_rule_gates_positive(self) -> None:
        result = evaluate_samples(self.positive, self.config)
        self.assertTrue(result.detected)
        self.assertEqual(result.roi_samples, 3)
        self.assertEqual(result.max_consecutive_samples, 3)

    def test_insufficient_dwell(self) -> None:
        self.assertFalse(evaluate_samples(self.positive[:2], self.config).detected)

    def test_insufficient_angle_change(self) -> None:
        samples = [WristSample(item.frame, item.x, item.y, 40.0, True) for item in self.positive]
        self.assertFalse(evaluate_samples(samples, self.config).detected)

    def test_insufficient_wrist_path(self) -> None:
        samples = [
            WristSample(0, .40, .50, 40.0, True), WristSample(10, .41, .50, 49.0, True),
            WristSample(20, .415, .50, 52.0, True),
        ]
        self.assertFalse(evaluate_samples(samples, self.config).detected)

    def test_no_wrist_detection(self) -> None:
        self.assertFalse(evaluate_samples([], self.config).detected)

    def test_invalid_roi_disables_with_warning(self) -> None:
        warnings: list[str] = []
        with patch.dict(os.environ, {"VISION_HANDLE_RULE": "1", "VISION_HANDLE_ROI": "bad"}, clear=False):
            self.assertIsNone(HandleRuleConfig.from_environment(warnings.append))
        self.assertTrue(warnings)

    def test_invalid_threshold_disables_with_warning(self) -> None:
        warnings: list[str] = []
        env = {"VISION_HANDLE_RULE": "1", "VISION_HANDLE_ROI": ".35,.44,.57,.68", "VISION_HANDLE_MIN_ROI_SAMPLES": "0"}
        with patch.dict(os.environ, env, clear=False):
            self.assertIsNone(HandleRuleConfig.from_environment(warnings.append))
        self.assertTrue(warnings)

    def test_a17_to_a18_swaps_scores(self) -> None:
        result = evaluate_samples(self.positive, self.config)
        probabilities = {"N1": .01, "A17": .80, "A18": .10, "A19": .03, "A20": 0., "A21": .06}
        prediction, event, changed, overridden = apply_a18_override("A17", "repeated_doorbell_press", probabilities, result)
        self.assertTrue(overridden)
        self.assertEqual(prediction, "A18")
        self.assertEqual(event, "door_lock_suspicious_activity")
        self.assertEqual(changed["A18"], .80)
        self.assertEqual(changed["A17"], .10)
        self.assertEqual(max(changed, key=changed.get), "A18")

    def test_existing_a18_and_other_classes_unchanged(self) -> None:
        result = evaluate_samples(self.positive, self.config)
        probabilities = {"N1": .2, "A17": .3, "A18": .4, "A19": .05, "A20": 0., "A21": .05}
        for prediction in ("A18", "N1", "A19", "A21"):
            output = apply_a18_override(prediction, "original", probabilities, result)
            self.assertEqual(output, (prediction, "original", probabilities, False))

    def test_non_detection_never_changes_a17(self) -> None:
        result = HandleRuleResult(False, 0, 0, 0, 0., 0.)
        probabilities = {"N1": .1, "A17": .8, "A18": .05, "A19": .02, "A20": 0., "A21": .03}
        self.assertEqual(apply_a18_override("A17", "event", probabilities, result), ("A17", "event", probabilities, False))


if __name__ == "__main__":
    unittest.main()
