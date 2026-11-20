from prometheus_client import Counter, Histogram, Gauge


# API 요청 횟수
# HTTP 메서드, 엔드포인트, 응답 상태 코드별 요청 수 측정
HTTP_REQUESTS = Counter(
    "audio_http_requests_total",
    "오디오 API의 전체 요청 횟수",
    ["method", "endpoint", "status_code"],
)


# API 응답 시간
# 요청부터 응답까지 걸린 시간을 초 단위로 측정
HTTP_DURATION = Histogram(
    "audio_http_duration_seconds",
    "오디오 API 요청의 전체 응답 시간(초)",
    ["endpoint"],
    buckets=(
        0.005, 0.01, 0.025, 0.05,
        0.1, 0.25, 0.5, 1, 2, 5, 10
    ),
)


# 오디오 처리 단계별 실행 시간
# FFmpeg 변환, 오디오 로딩, Mel 전처리, CNN 추론 시간 측정
STAGE_DURATION = Histogram(
    "audio_stage_duration_seconds",
    "오디오 처리 단계별 실행 시간(초)",
    ["stage"],
    buckets=(
        0.001, 0.005, 0.01, 0.025,
        0.05, 0.1, 0.25, 0.5, 1, 2, 5
    ),
)


# 요청당 처리한 오디오 윈도우 수
# 조기 종료 여부에 따른 처리량 차이 분석
WINDOWS_PROCESSED = Histogram(
    "audio_windows_processed",
    "오디오 예측 요청당 처리한 윈도우 개수",
    buckets=(1, 2, 5, 10, 15, 20, 25, 50, 100),
)


# 오디오 예측 결과별 횟수
# background, knock, handle 클래스별 예측 횟수 측정
PREDICTIONS = Counter(
    "audio_predictions_total",
    "오디오 이벤트 클래스별 예측 횟수",
    ["predicted_class"],
)


# 현재 처리 중인 요청 수
# 동시 요청 부하 및 서버 처리 상태 확인
IN_PROGRESS = Gauge(
    "audio_requests_in_progress",
    "현재 처리 중인 오디오 예측 요청 수",
)