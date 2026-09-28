import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.config import KinematicConfig
from src.models import KinematicPhase
from src.spotting.kinematic_spotter import KinematicSpotter


def test_wrist_extraction():
    spotter = KinematicSpotter()
    features = np.zeros(118, dtype=np.float32)

    # Задаем координаты точки 0 левой кисти MediaPipe (индексы 34, 35)
    features[34] = 120.0
    features[35] = 250.0

    # Задаем координаты точки 10 правой кисти YOLO (индексы 20, 21)
    features[20] = 300.0
    features[21] = 400.0

    lw, rw = spotter._extract_wrists(features)
    assert lw == (120.0, 250.0)
    assert rw == (300.0, 400.0)


def test_ema_smoothing():
    config = KinematicConfig(ema_alpha=0.3)
    spotter = KinematicSpotter(config=config)

    features1 = np.zeros(118, dtype=np.float32)
    features1[34] = 100.0
    features1[35] = 100.0
    state1, _ = spotter.process_frame(features1, timestamp=0.0)

    # Кадр 2: резкий сдвиг на 10 пикселей
    features2 = np.zeros(118, dtype=np.float32)
    features2[34] = 110.0
    features2[35] = 100.0
    state2, _ = spotter.process_frame(features2, timestamp=0.033)

    # v_raw = 10.0, v_ema = 0.3 * 10.0 + 0.7 * 0 = 3.0
    assert abs(state2.v_raw - 10.0) < 1e-4
    assert abs(state2.v_ema - 3.0) < 1e-4


def test_uniform_16_sampling():
    spotter = KinematicSpotter()
    # Буфер из 40 кадров
    buffer = [np.full(118, fill_value=i, dtype=np.float32) for i in range(40)]
    sampled = spotter._sample_uniform_16(buffer)

    assert sampled.shape == (16, 118)
    # Первый кадр должен быть из начала, последний из конца
    assert sampled[0, 0] == 0.0
    assert sampled[-1, 0] == 39.0


def test_phase_transition_to_boundary():
    config = KinematicConfig(
        velocity_threshold_active=10.0,
        velocity_threshold_stop=5.0,
        acceleration_threshold_stop=-2.0,
        pause_frames_trigger=3,
        min_gesture_frames=5
    )
    spotter = KinematicSpotter(config=config)

    # 1. Покой (IDLE)
    for t in range(5):
        feat = np.zeros(118, dtype=np.float32)
        feat[34] = 100.0
        state, seg = spotter.process_frame(feat, timestamp=t * 0.033)
        assert state.phase == KinematicPhase.IDLE
        assert seg is None

    # 2. Активное движение (PREPARATION -> STROKE)
    curr_x = 100.0
    for t in range(5, 18):
        curr_x += 25.0  # Быстрое перемещение
        feat = np.zeros(118, dtype=np.float32)
        feat[34] = curr_x
        state, seg = spotter.process_frame(feat, timestamp=t * 0.033)

    assert state.phase in (KinematicPhase.STROKE, KinematicPhase.RETRACTION)

    # 3. Остановка (RETRACTION -> TRIGGER)
    segment_triggered = None
    for t in range(18, 30):
        # Рука остановилась
        feat = np.zeros(118, dtype=np.float32)
        feat[34] = curr_x
        state, seg = spotter.process_frame(feat, timestamp=t * 0.033)
        if seg is not None:
            segment_triggered = seg
            break

    assert segment_triggered is not None
    assert segment_triggered.tensor.shape == (16, 118)
    assert segment_triggered.segment_id == 1
    print("[PASS] test_kinematic_spotter: all assertions passed!")


if __name__ == "__main__":
    test_wrist_extraction()
    test_ema_smoothing()
    test_uniform_16_sampling()
    test_phase_transition_to_boundary()

