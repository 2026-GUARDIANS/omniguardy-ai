
"""3-class Keras 모델 로딩과 추론을 캡슐화합니다."""

from threading import Lock

import numpy as np
import tensorflow as tf


class KerasModelRepository:
    """설정 경로에서 Keras 모델을 한 번 로드해 재사용합니다."""

    def __init__(self, model_path):
        self.model_path = str(model_path)
        self.model = tf.keras.models.load_model(
            self.model_path, compile=False
        )
        self._predict_lock = Lock()

    def predict_probabilities(self, model_input):
        with self._predict_lock:
            probabilities = np.asarray(
                self.model.predict(model_input, verbose=0),
                dtype=np.float32,
            )

        if probabilities.ndim != 2 or probabilities.shape[1] != 3:
            raise ValueError(
                "Door Event model must return (windows, 3) softmax probabilities"
            )

        return probabilities
