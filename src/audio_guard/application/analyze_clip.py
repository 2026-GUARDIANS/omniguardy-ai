
"""오디오 클립을 window 순서대로 분석해 즉시 Door Event를 반환합니다."""

import logging
from time import perf_counter

import numpy as np

from audio_guard.domain.event_policy import CooldownGate, EventPolicy
from audio_guard.domain.events import DoorEventResult
from audio_guard.domain.pipeline import create_pipeline
from audio_guard.infrastructure.monitoring.metrics import (
    STAGE_DURATION,
    WINDOWS_PROCESSED,
    PREDICTIONS,
)
from audio_guard.labels import CLASS_NAMES

logger = logging.getLogger(__name__)


class AnalyzeClip:
    """첫 threshold 초과 window를 이벤트로 변환합니다."""

    def __init__(
        self,
        model_repository,
        pipeline_name,
        thresholds,
        window_seconds=1.0,
        hop_seconds=0.2,
        cooldown_seconds=1.0,
        cooldown_gate=None,
    ):
        self.model_repository = model_repository
        self.pipeline = create_pipeline(
            pipeline_name,
            window_seconds,
            hop_seconds,
        )
        self.policy = EventPolicy(thresholds)
        self.cooldown = cooldown_gate or CooldownGate(
            cooldown_seconds
        )

    def execute(self, clip):
        best_background = None
        suppressed = False
        first_offset = 0.0

        windows_count = 0

        try:
            for window_input, offset in self.pipeline.iter_features(
                clip.samples,
                clip.sample_rate,
            ):
                windows_count += 1

                # CNN 추론 시간 측정
                start = perf_counter()

                try:
                    probabilities = (
                        self.model_repository.predict_probabilities(
                            window_input[None, ...]
                        )[0]
                    )
                finally:
                    STAGE_DURATION.labels(
                        stage="cnn_inference"
                    ).observe(perf_counter() - start)

                predicted_class, status = self.policy.decide(
                    probabilities
                )

                if predicted_class == "background":
                    if (
                        best_background is None
                        or probabilities[0] > best_background[0][0]
                    ):
                        best_background = (probabilities, offset)
                    continue

                if not self.cooldown.allow():
                    suppressed = True
                    continue

                result = self._result(
                    status,
                    predicted_class,
                    probabilities,
                    offset,
                )

                PREDICTIONS.labels(
                    predicted_class=result.predicted_class
                ).inc()

                logger.info(
                    "Door event: status=%s class=%s "
                    "window_start=%.3f probabilities=%s",
                    result.status,
                    result.predicted_class,
                    result.window_start_seconds,
                    result.probabilities,
                )

                return result

            if best_background is None:
                best_background = (
                    np.array(
                        [1.0, 0.0, 0.0],
                        dtype=np.float32,
                    ),
                    first_offset,
                )

            probabilities, offset = best_background

            result = self._result(
                "NO_EVENT",
                "background",
                probabilities,
                offset,
                cooldown_suppressed=suppressed,
            )

            PREDICTIONS.labels(
                predicted_class=result.predicted_class
            ).inc()

            return result

        finally:
            # Early Exit 또는 예외 발생 시에도 처리한 윈도우 수 기록
            WINDOWS_PROCESSED.observe(windows_count)

    @staticmethod
    def _result(
        status,
        predicted_class,
        probabilities,
        offset,
        cooldown_suppressed=False,
    ):
        return DoorEventResult(
            status=status,
            predicted_class=predicted_class,
            probabilities={
                name: float(probabilities[index])
                for index, name in enumerate(CLASS_NAMES)
            },
            window_start_seconds=float(offset),
            cooldown_suppressed=cooldown_suppressed,
        )
