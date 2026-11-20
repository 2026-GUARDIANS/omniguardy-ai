
"""오디오 추론 중 Health API 응답성을 비교하는 실험."""

import argparse
import csv
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Event

import requests

BASE_URL = "http://127.0.0.1:8000"
AUDIO_FILE = Path("benchmark_samples/background_5s.wav")


def send_predict(request_id, start_event):
    """오디오 분석 요청과 실행 구간을 기록합니다."""
    start_event.wait()
    started = time.perf_counter()

    try:
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

        success = True
        error = ""

    except Exception as exc:
        success = False
        error = str(exc)

    ended = time.perf_counter()

    return {
        "type": "predict",
        "request_id": request_id,
        "started": started,
        "ended": ended,
        "latency_ms": (ended - started) * 1000,
        "success": success,
        "error": error,
    }


def send_health(request_id, start_event):
    """Health API 요청의 시작과 종료 시각을 기록합니다."""
    start_event.wait()
    started = time.perf_counter()

    try:
        response = requests.get(
            f"{BASE_URL}/health",
            timeout=30,
        )
        response.raise_for_status()
        success = True
        error = ""

    except Exception as exc:
        success = False
        error = str(exc)

    ended = time.perf_counter()

    return {
        "type": "health",
        "request_id": request_id,
        "started": started,
        "ended": ended,
        "latency_ms": (ended - started) * 1000,
        "success": success,
        "error": error,
    }


def percentile(values, p):
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * p) - 1)
    return ordered[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--predict-concurrency", type=int, default=4)
    parser.add_argument("--health-requests", type=int, default=20)
    args = parser.parse_args()

    if not AUDIO_FILE.is_file():
        raise FileNotFoundError(AUDIO_FILE)

    requests.get(
        f"{BASE_URL}/health",
        timeout=10,
    ).raise_for_status()

    start_event = Event()

    with ThreadPoolExecutor(
        max_workers=(
            args.predict_concurrency
            + args.health_requests
        )
    ) as executor:

        predict_futures = [
            executor.submit(send_predict, i + 1, start_event)
            for i in range(args.predict_concurrency)
        ]

        health_futures = [
            executor.submit(send_health, i + 1, start_event)
            for i in range(args.health_requests)
        ]

        # 모든 요청을 최대한 동시에 시작합니다.
        start_event.set()

        predict_results = [
            future.result()
            for future in as_completed(predict_futures)
        ]

        health_results = [
            future.result()
            for future in as_completed(health_futures)
        ]

    # Health 요청 시작 시점에 Predict 요청이 진행 중이었는지 확인
    for health in health_results:
        health["overlap"] = any(
            predict["started"] <= health["started"] < predict["ended"]
            for predict in predict_results
        )

    overlapping = [
        result
        for result in health_results
        if result["overlap"] and result["success"]
    ]

    predict_failures = sum(
        not result["success"]
        for result in predict_results
    )

    health_failures = sum(
        not result["success"]
        for result in health_results
    )

    print(f"\n=== {args.label} ===")
    print(f"Predict requests : {len(predict_results)}")
    print(f"Predict failures : {predict_failures}")
    print(f"Health requests  : {len(health_results)}")
    print(f"Health failures  : {health_failures}")
    print(f"Overlapping      : {len(overlapping)}")

    if overlapping:
        values = [
            result["latency_ms"]
            for result in overlapping
        ]

        print(
            f"Health avg       : "
            f"{statistics.mean(values):.2f} ms"
        )
        print(
            f"Health P95       : "
            f"{percentile(values, 0.95):.2f} ms"
        )
        print(
            f"Health max       : "
            f"{max(values):.2f} ms"
        )

    output = Path(f"health_overlap_{args.label}.csv")

    rows = predict_results + health_results

    with output.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "type",
                "request_id",
                "started",
                "ended",
                "latency_ms",
                "success",
                "error",
                "overlap",
            ],
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
