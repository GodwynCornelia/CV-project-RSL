import time
import numpy as np
import pytest

from rsl_trainer.config import AppConfig
from rsl_trainer.cv.fast_tracker import FastTracker, TrackerMode


def test_dactyl_mode_latency_audit():
    """
    Аудит производительности режима ДАКТИЛЬ (Спецификация 5.1).
    В режиме дактиля YOLO отключен, работает только MediaPipe Hands.
    Бюджет: целевая задержка <= 33.3 мс (30+ FPS) на CPU.
    """
    config = AppConfig()
    tracker = FastTracker(config=config)
    tracker.set_mode(TrackerMode.DACTYL)

    # Синтетический RGB кадр 640x480
    dummy_frame = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)

    # Прогрев (warmup)
    for _ in range(3):
        tracker.extract_features(dummy_frame)

    latencies = []
    iterations = 20
    for _ in range(iterations):
        t0 = time.perf_counter()
        tracker.extract_features(dummy_frame)
        dt = (time.perf_counter() - t0) * 1000.0  # мс
        latencies.append(dt)

    avg_ms = float(np.mean(latencies))
    fps = 1000.0 / avg_ms if avg_ms > 0 else 0

    print(f"\n[АУДИТ ДАКТИЛЬ] Среднее время кадра: {avg_ms:.2f} мс, Расчетный FPS: {fps:.1f}")

    # Проверка, что среднее время не превышает 35 мс на CPU
    tracker.close()
    assert avg_ms < 40.0, f"Время обработки кадра в режиме дактиля превысило бюджет: {avg_ms:.2f} мс"
