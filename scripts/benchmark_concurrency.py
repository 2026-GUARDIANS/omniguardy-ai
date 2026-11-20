
import csv
import math
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

URL = "http://127.0.0.1:8000/predict"
AUDIO_FILE = Path("benchmark_samples/background_5s.wav")

CONCURRENCY_LEVELS = [1, 2, 4]
REQUESTS_PER_LEVEL = 12
TIMEOUT = 60


def send_request(request_id):
    start = time.perf_counter()

    try:
        with AUDIO_FILE.open("rb") as audio:
            response = requests.post(
                URL,
                files={"file": (AUDIO_FILE.name, audio, "audio/wav")},
                timeout=TIMEOUT,
            )

        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "request_id": request_id,
            "latency_ms": round(elapsed_ms, 2),
            "success": response.status_code == 200,
            "status_code": response.status_code,
        }

    except requests.RequestException:
        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "request_id": request_id,
            "latency_ms": round(elapsed_ms, 2),
            "success": False,
            "status_code": "ERROR",
        }


def percentile(values, p):
    values = sorted(values)
    index = math.ceil(len(values) * p) - 1
    return values[max(0, index)]


def main():
    if not AUDIO_FILE.is_file():
        raise FileNotFoundError(AUDIO_FILE)

    all_results = []

    for concurrency in CONCURRENCY_LEVELS:
        print(f"\n=== Concurrency {concurrency} ===")

        start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(send_request, i + 1)
                for i in range(REQUESTS_PER_LEVEL)
            ]

            results = [
                future.result()
                for future in as_completed(futures)
            ]

        total_seconds = time.perf_counter() - start

        for result in results:
            result["concurrency"] = concurrency

        all_results.extend(results)

        successful = [
            r["latency_ms"]
            for r in results
            if r["success"]
        ]

        failures = sum(not r["success"] for r in results)
        throughput = len(results) / total_seconds

        print(f"Requests       : {len(results)}")
        print(f"Failures       : {failures}")
        print(f"Total time     : {total_seconds:.2f} s")
        print(f"Throughput     : {throughput:.2f} req/s")

        if successful:
            print(f"Average latency: {statistics.mean(successful):.2f} ms")
            print(f"P95 latency    : {percentile(successful, 0.95):.2f} ms")
            print(f"Max latency    : {max(successful):.2f} ms")

        time.sleep(3)

    output = Path("concurrency_results.csv")

    with output.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "request_id",
                "latency_ms",
                "success",
                "status_code",
                "concurrency",
            ],
        )
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
