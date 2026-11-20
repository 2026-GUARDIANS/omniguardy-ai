
import os
import time
import statistics
import psutil
import requests

URL = os.getenv("AUDIO_URL", "http://127.0.0.1:8000/predict")
FILE_PATH = os.getenv("AUDIO_FILE", "test_audio.wav")
DURATION = int(os.getenv("DURATION", "600"))
INTERVAL = float(os.getenv("INTERVAL", "1.0"))

process = psutil.Process()
cpu_samples = []
memory_samples = []
latencies = []
failures = 0
requests_sent = 0

# FastAPI 프로세스가 아닌 별도 클라이언트에서 실행하므로
# 이 스크립트는 로컬 시스템 전체 CPU를 측정합니다.
psutil.cpu_percent(interval=None)

start = time.perf_counter()

while time.perf_counter() - start < DURATION:
    request_start = time.perf_counter()

    try:
        with open(FILE_PATH, "rb") as audio:
            response = requests.post(
                URL,
                files={"file": audio},
                timeout=30,
            )
        response.raise_for_status()
        if response.json().get("status", "").startswith("error"):
            failures += 1
    except (requests.RequestException, ValueError):
        failures += 1

    requests_sent += 1
    latencies.append(
        (time.perf_counter() - request_start) * 1000
    )
    cpu_samples.append(psutil.cpu_percent(interval=None))
    memory_samples.append(psutil.virtual_memory().percent)

    remaining = INTERVAL - (
        time.perf_counter() - request_start
    )
    if remaining > 0:
        time.sleep(remaining)

print("요청 수:", requests_sent)
print("실패 수:", failures)
print("시스템 평균 CPU(%):", round(statistics.mean(cpu_samples), 2))
print("시스템 최대 CPU(%):", max(cpu_samples))
print("시스템 평균 메모리 사용률(%):",
      round(statistics.mean(memory_samples), 2))
print("평균 응답 시간(ms):",
      round(statistics.mean(latencies), 2))
print("최대 응답 시간(ms):", round(max(latencies), 2))
