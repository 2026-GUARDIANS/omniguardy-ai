
import csv
import statistics
import time
from pathlib import Path

import requests

URL = "http://127.0.0.1:8000/predict"

# 실제 테스트 오디오 파일 경로로 변경
AUDIO_FILE = Path("benchmark_samples/background_5s.wav")

WARMUP = 5
MEASUREMENTS = 30

results = []

if not AUDIO_FILE.is_file():
    raise FileNotFoundError(
        f"오디오 파일을 찾을 수 없습니다: {AUDIO_FILE.resolve()}"
    )

for i in range(WARMUP + MEASUREMENTS):
    with AUDIO_FILE.open("rb") as audio:
        start = time.perf_counter()

        response = requests.post(
            URL,
            files={"file": (AUDIO_FILE.name, audio)},
            timeout=60,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

    response.raise_for_status()
    prediction = response.json()

    phase = "WARMUP" if i < WARMUP else "MEASURE"

    print(
        f"[{phase}] {i + 1:02d} | "
        f"{elapsed_ms:.2f} ms | "
        f"{prediction.get('predicted_class')}"
    )

    if i >= WARMUP:
        results.append({
            "iteration": i - WARMUP + 1,
            "response_ms": elapsed_ms,
            "predicted_class": prediction.get("predicted_class"),
            "status": prediction.get("status"),
        })

    # 서버에 연속 요청이 과도하게 쌓이지 않도록 간격 유지
    time.sleep(0.2)

output = Path("benchmark_results.csv")

with output.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "iteration",
            "response_ms",
            "predicted_class",
            "status",
        ],
    )
    writer.writeheader()
    writer.writerows(results)

times = sorted(row["response_ms"] for row in results)
p95_index = max(0, int(0.95 * len(times) + 0.999999) - 1)

print("\n========== BENCHMARK RESULT ==========")
print(f"Measured requests : {len(times)}")
print(f"Average latency   : {statistics.mean(times):.2f} ms")
print(f"Median latency    : {statistics.median(times):.2f} ms")
print(f"P95 latency       : {times[p95_index]:.2f} ms")
print(f"Min latency       : {min(times):.2f} ms")
print(f"Max latency       : {max(times):.2f} ms")
print(f"Saved CSV         : {output.resolve()}")
