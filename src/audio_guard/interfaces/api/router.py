
"""오디오 예측과 상태 확인 FastAPI 라우터입니다."""

import logging
import os
import tempfile
from time import perf_counter

from fastapi import APIRouter, File, UploadFile

from audio_guard.domain.audio_clip import AudioClip
from audio_guard.infrastructure.audio.librosa_loader import load_audio
from audio_guard.infrastructure.monitoring.metrics import STAGE_DURATION
from audio_guard.interfaces.api.schemas import PredictResponse

logger = logging.getLogger(__name__)


def create_router(analyze_clip, converter, config):
    router = APIRouter()

    @router.post("/predict", response_model=PredictResponse)
    async def predict(file: UploadFile = File(...)):
        logger.info("REQUEST RECEIVED : %s", file.filename)

        tmp_path = None
        wav_path = None

        try:
            # 1. 업로드 파일 읽기
            contents = await file.read()

            if not contents:
                return {"status": "error_empty_file"}

            # 2. 임시 파일 저장
            suffix = os.path.splitext(file.filename or "")[1] or ".wav"

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as temporary:
                temporary.write(contents)
                tmp_path = temporary.name

            wav_path = os.path.splitext(tmp_path)[0] + "_converted.wav"

            # 3. FFmpeg 변환 시간 측정
            start = perf_counter()

            try:
                converted = converter.convert_to_wav(
                    tmp_path,
                    wav_path,
                )
            finally:
                STAGE_DURATION.labels(
                    stage="ffmpeg"
                ).observe(perf_counter() - start)

            if not converted:
                return {"status": "error_ffmpeg"}

            # 4. Librosa 로딩 시간 측정
            start = perf_counter()

            try:
                audio, sample_rate = load_audio(
                    wav_path,
                    config["sample_rate"],
                )
            finally:
                STAGE_DURATION.labels(
                    stage="audio_load"
                ).observe(perf_counter() - start)

            # 5. AudioClip 생성
            clip = AudioClip(audio, sample_rate)

            if clip.duration < 1.0:
                return {"status": "error_short_audio"}

            # 6. 기존 CNN 분석 실행
            result = analyze_clip.execute(clip)

            # 7. 기존 API 응답 유지
            return {
                "status": result.status,
                "predicted_class": result.predicted_class,
                "probabilities": result.probabilities,
                "window_start_seconds": result.window_start_seconds,
                "cooldown_suppressed": result.cooldown_suppressed,
            }

        except Exception as error:
            logger.exception("API ERROR : %s", error)
            return {"status": "error"}

        finally:
            # 임시 파일 정리
            for path in (tmp_path, wav_path):
                if path and os.path.exists(path):
                    os.remove(path)

    @router.get("/health")
    def health():
        return {
            "status": "ok",
            "model": os.path.basename(config["model_path"]),
            "pipeline": config["pipeline"],
            "thresholds": config["thresholds"],
            "sample_rate": config["sample_rate"],
            "window_seconds": config["window_seconds"],
            "hop_seconds": config["hop_seconds"],
            "cooldown_seconds": config["cooldown_seconds"],
            "ffmpeg": converter.executable,
        }

    return router
