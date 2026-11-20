
"""오디오 분석 중 /health 응답성을 측정합니다."""

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Event

import requests

BASE_URL = "http://127.0.0.1:8000"
AUDIO_FILE = Path("benchmark_samples/background_5s.wav")


def predict(start_event):
    """동시 오디오 분석 요청을 전송합니다."""
    start_event.wait()

    with AUDIO_FILE.open("rb") as audio:
        response = requests.post(
            f"{BASE_URL}/predict",
            files={
                "file": (
                    AUDIO_FILE.name,
                    audio,
                    "audio/wav",
                )
            },
            timeout=60,
        )

    response.raise_for_status()
    result = response.json()

    if result.get("status", "").startswith("error"):
        raise RuntimeError(f"Predict failed: {result}")

    return result


def measure_health():
    """Health API의 응답 시간을 측정합니다."""
    start = time.perf_counter()

    response = requests.get(
        f"{BASE_URL}/health",
        timeout=30,
    )

    elapsed_ms = (time.perf_counter() - start) * 1000
    response.raise_for_status()

    return elapsed_ms


def percentile(values, p):
    values = sorted(values)
    index = max(0, int(len(values) * p + 0.999999) - 1)
    return values[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--predict-concurrency", type=int, default=4)
    parser.add_argument("--health-requests", type=int, default=20)
    args = parser.parse_args()

    if not AUDIO_FILE.is_file():
        raise FileNotFoundError(AUDIO_FILE)

    # 서버가 정상인지 사전 확인
    requests.get(f"{BASE_URL}/health", timeout=10).raise_for_status()

    start_event = Event()
    health_latencies = []

    with ThreadPoolExecutor(
        max_workers=args.predict_concurrency + 1
    ) as executor:
        predict_futures = [
            executor.submit(predict, start_event)
            for _ in range(args.predict_concurrency)
        ]

        # 분석 요청을 동시에 시작
        start_event.set()
        time.sleep(0.1)

        for _ in range(args.health_requests):
            health_latencies.append(measure_health())

        # 분석 요청 결과까지 확인
        for future in as_completed(predict_futures):
            future.result()

    print(f"\n=== {args.label} ===")
    print(f"Predict concurrency: {args.predict_concurrency}")
    print(f"Health requests    : {len(health_latencies)}")
    print(f"Health avg         : {statistics.mean(health_latencies):.2f} ms")
    print(f"Health P95         : {percentile(health_latencies, 0.95):.2f} ms")
    print(f"Health max         : {max(health_latencies):.2f} ms")


if __name__ == "__main__":
    main()
