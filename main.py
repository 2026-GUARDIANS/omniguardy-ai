
"""FastAPI 애플리케이션 의존성을 조립합니다."""

import logging
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, Request
from prometheus_client import make_asgi_app

from audio_guard.application.analyze_clip import AnalyzeClip
from audio_guard.config import load_config
from audio_guard.infrastructure.audio.ffmpeg_converter import FfmpegConverter
from audio_guard.infrastructure.ml.keras_model_repository import KerasModelRepository
from audio_guard.infrastructure.monitoring.metrics import (
    HTTP_REQUESTS,
    HTTP_DURATION,
    IN_PROGRESS,
)
from audio_guard.interfaces.api.router import create_router


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# =============================
# 기존 설정 로딩
# =============================

config = load_config()

logger.info(
    "Door Event runtime: model=%s sample_rate=%s "
    "thresholds=%s window=%s hop=%s cooldown=%s",
    Path(config["model_path"]).name,
    config["sample_rate"],
    config["thresholds"],
    config["window_seconds"],
    config["hop_seconds"],
    config["cooldown_seconds"],
)


# =============================
# 기존 모델 및 의존성 생성
# =============================

model_repository = KerasModelRepository(
    config["model_path"]
)

converter = FfmpegConverter(
    config["sample_rate"]
)

analyze_clip = AnalyzeClip(
    model_repository,
    config["pipeline"],
    config["thresholds"],
    config["window_seconds"],
    config["hop_seconds"],
    config["cooldown_seconds"],
)


# =============================
# FastAPI 앱 생성
# =============================

app = FastAPI()


# =============================
# Prometheus 엔드포인트
# =============================

# Prometheus가 성능 지표를 수집할 수 있도록 노출
app.mount("/metrics", make_asgi_app())


# =============================
# Prometheus HTTP Middleware
# =============================

@app.middleware("http")
async def metrics_middleware(
    request: Request,
    call_next,
):
    """
    /predict API의 요청 횟수,
    응답 시간, 처리 중인 요청 수를 측정합니다.
    """

    # 예측 API 외의 요청은 측정하지 않음
    if request.url.path != "/predict":
        return await call_next(request)

    start = perf_counter()
    status_code = 500

    IN_PROGRESS.inc()

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response

    finally:
        duration = perf_counter() - start

        HTTP_REQUESTS.labels(
            method=request.method,
            endpoint="/predict",
            status_code=str(status_code),
        ).inc()

        HTTP_DURATION.labels(
            endpoint="/predict",
        ).observe(duration)

        IN_PROGRESS.dec()


# =============================
# 기존 API Router 등록
# =============================

app.include_router(
    create_router(
        analyze_clip,
        converter,
        config,
    )
)
