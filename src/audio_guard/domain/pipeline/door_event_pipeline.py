
"""1초 sliding window 기반 Door Event 전처리입니다."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from audio_guard.domain.pipeline.base import (
    WINDOW_HOP_SECONDS,
    WINDOW_SECONDS,
)
from audio_guard.domain.pipeline.mel_spectrogram import audio_to_mel
from audio_guard.infrastructure.monitoring.metrics import STAGE_DURATION


@dataclass(frozen=True)
class AudioWindow:
    start_seconds: float
    samples: np.ndarray


def audio_windows(
    audio,
    sample_rate,
    window_seconds=WINDOW_SECONDS,
    hop_seconds=WINDOW_HOP_SECONDS,
):
    window_samples = int(round(window_seconds * sample_rate))
    hop_samples = int(round(hop_seconds * sample_rate))

    if window_samples <= 0 or hop_samples <= 0:
        raise ValueError("Window and hop must be positive")

    if hop_samples > window_samples:
        raise ValueError("Hop must not exceed window")

    audio = np.asarray(audio, dtype=np.float32)

    if len(audio) <= window_samples:
        yield AudioWindow(
            0.0,
            np.pad(
                audio,
                (0, max(0, window_samples - len(audio))),
            ),
        )
        return

    last_start = len(audio) - window_samples
    starts = list(range(0, last_start + 1, hop_samples))

    if starts[-1] != last_start:
        starts.append(last_start)

    for start in starts:
        yield AudioWindow(
            start / sample_rate,
            audio[start:start + window_samples],
        )


class DoorEventPipeline:
    name = "door_event"

    def __init__(
        self,
        window_seconds=WINDOW_SECONDS,
        hop_seconds=WINDOW_HOP_SECONDS,
    ):
        if (
            window_seconds <= 0
            or hop_seconds <= 0
            or hop_seconds > window_seconds
        ):
            raise ValueError("Invalid window/hop configuration")

        self.window_seconds = float(window_seconds)
        self.hop_seconds = float(hop_seconds)

    def iter_features(self, audio, sample_rate):
        for window in audio_windows(
            audio,
            sample_rate,
            self.window_seconds,
            self.hop_seconds,
        ):
            # Mel Spectrogram 특징 추출 시간 측정
            start = perf_counter()

            try:
                mel_features = audio_to_mel(
                    window.samples,
                    sample_rate,
                )
            finally:
                STAGE_DURATION.labels(
                    stage="mel_preprocessing"
                ).observe(perf_counter() - start)

            yield (
                mel_features[..., None],
                window.start_seconds,
            )

    def transform_with_offsets(self, audio, sample_rate):
        transformed = list(
            self.iter_features(audio, sample_rate)
        )

        features = np.stack(
            [feature for feature, _ in transformed]
        )

        offsets = np.asarray(
            [offset for _, offset in transformed],
            dtype=np.float32,
        )

        return features, offsets

    def transform(self, audio, sample_rate):
        features, _ = self.transform_with_offsets(
            audio,
            sample_rate,
        )
        return features
