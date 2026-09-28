import math
from typing import List, Optional, Tuple
import numpy as np

from src.config import KinematicConfig
from src.models import GestureSegment, KinematicPhase, KinematicState


class KinematicSpotter:
    """
    Патентоспособный модуль детекции фаз и границ жестов (Action Spotting).
    Анализирует дифференциальные скорости и фазовые остановки траекторий запястий.
    Сглаживает мгновенную кинематику через EMA (alpha=0.3) и генерирует
    16-кадровые тензоры жестов (16, 118) в моменты граничных триггеров.
    """

    def __init__(self, config: KinematicConfig = KinematicConfig()):
        self.config = config

        self._frame_idx: int = 0
        self._segment_counter: int = 0

        # Состояния кинематики
        self._prev_left_wrist: Optional[Tuple[float, float]] = None
        self._prev_right_wrist: Optional[Tuple[float, float]] = None
        self._prev_v_ema: float = 0.0
        self._current_phase: KinematicPhase = KinematicPhase.IDLE

        # Внутренние счетчики фаз
        self._pause_counter: int = 0
        self._peak_velocity: float = 0.0

        # Буфер кадров текущего формируемого жеста (каждый кадр: 118 фичей)
        self._segment_buffer: List[np.ndarray] = []
        self._segment_start_time: float = 0.0

    @staticmethod
    def _extract_wrists(features_118: np.ndarray) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        """
        Извлекает координаты запястий из 118-мерного вектора признаков.
        Приоритет отдается MediaPipe Hand 0 (основание кисти), затем YOLO (точки 9 и 10).
        """
        # 1. MediaPipe основание левой кисти (индексы 34, 35)
        mp_lw = (float(features_118[34]), float(features_118[35]))
        # 2. MediaPipe основание правой кисти (индексы 76, 77)
        mp_rw = (float(features_118[76]), float(features_118[77]))

        # 3. YOLO запястья: точка 9 (индексы 18, 19), точка 10 (индексы 20, 21)
        yolo_lw = (float(features_118[18]), float(features_118[19]))
        yolo_rw = (float(features_118[20]), float(features_118[21]))

        left_wrist = mp_lw if (mp_lw[0] != 0 or mp_lw[1] != 0) else (yolo_lw if (yolo_lw[0] != 0 or yolo_lw[1] != 0) else None)
        right_wrist = mp_rw if (mp_rw[0] != 0 or mp_rw[1] != 0) else (yolo_rw if (yolo_rw[0] != 0 or yolo_rw[1] != 0) else None)

        return left_wrist, right_wrist

    @staticmethod
    def _euclidean_dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def process_frame(
        self,
        features_118: np.ndarray,
        timestamp: float
    ) -> Tuple[KinematicState, Optional[GestureSegment]]:
        """
        Обрабатывает один кадр видеопотока.
        Возвращает метрики кинематики и (при фиксации границы) готовый GestureSegment.
        """
        self._frame_idx += 1
        left_wrist, right_wrist = self._extract_wrists(features_118)

        # 1. Вычисление мгновенных скоростей кистей рук
        v_left = 0.0
        v_right = 0.0

        if left_wrist is not None and self._prev_left_wrist is not None:
            v_left = self._euclidean_dist(left_wrist, self._prev_left_wrist)
        if right_wrist is not None and self._prev_right_wrist is not None:
            v_right = self._euclidean_dist(right_wrist, self._prev_right_wrist)

        self._prev_left_wrist = left_wrist
        self._prev_right_wrist = right_wrist

        v_raw = max(v_left, v_right)

        # 2. Экспоненциальное сглаживание скорости (EMA, alpha=0.3)
        alpha = self.config.ema_alpha
        v_ema = alpha * v_raw + (1.0 - alpha) * self._prev_v_ema

        # 3. Дифференциальное ускорение (a_t = v_t - v_{t-1})
        a_t = v_ema - self._prev_v_ema
        self._prev_v_ema = v_ema

        # 4. Конечный автомат фаз жеста (Finite State Machine)
        is_boundary = False
        completed_segment: Optional[GestureSegment] = None

        if self._current_phase == KinematicPhase.IDLE:
            if v_ema >= self.config.velocity_threshold_active:
                self._current_phase = KinematicPhase.PREPARATION
                self._segment_buffer = [features_118.copy()]
                self._segment_start_time = timestamp
                self._peak_velocity = v_ema
                self._pause_counter = 0

        elif self._current_phase == KinematicPhase.PREPARATION:
            self._segment_buffer.append(features_118.copy())
            if v_ema > self._peak_velocity:
                self._peak_velocity = v_ema

            if v_ema > self.config.velocity_threshold_active * 1.2:
                self._current_phase = KinematicPhase.STROKE
            elif v_ema < self.config.velocity_threshold_stop and a_t <= self.config.acceleration_threshold_stop:
                # Ложное срабатывание или микро-движение
                if len(self._segment_buffer) < self.config.min_gesture_frames:
                    self._current_phase = KinematicPhase.IDLE
                    self._segment_buffer.clear()

        elif self._current_phase == KinematicPhase.STROKE:
            self._segment_buffer.append(features_118.copy())
            if v_ema > self._peak_velocity:
                self._peak_velocity = v_ema

            # Переход в ретракцию при резком торможении или спаде скорости
            if a_t <= self.config.acceleration_threshold_stop or v_ema < self.config.velocity_threshold_stop:
                self._current_phase = KinematicPhase.RETRACTION
                self._pause_counter = 1

        elif self._current_phase == KinematicPhase.RETRACTION:
            self._segment_buffer.append(features_118.copy())

            if v_ema < self.config.velocity_threshold_stop:
                self._pause_counter += 1
            else:
                self._pause_counter = 0

            # Условие фиксации границы (Boundary Trigger):
            # устойчивая остановка либо превышение максимального окна
            buffer_len = len(self._segment_buffer)
            reached_pause = self._pause_counter >= self.config.pause_frames_trigger
            exceeded_max = buffer_len >= self.config.max_gesture_frames

            if reached_pause or exceeded_max:
                is_boundary = True
                self._current_phase = KinematicPhase.TRIGGER

                if buffer_len >= self.config.min_gesture_frames:
                    self._segment_counter += 1
                    sampled_tensor = self._sample_uniform_16(self._segment_buffer)
                    completed_segment = GestureSegment(
                        segment_id=self._segment_counter,
                        start_time=self._segment_start_time,
                        end_time=timestamp,
                        frame_count=buffer_len,
                        tensor=sampled_tensor
                    )

                # Сброс буфера для следующего жеста
                self._segment_buffer.clear()
                self._current_phase = KinematicPhase.IDLE
                self._pause_counter = 0
                self._peak_velocity = 0.0

        state = KinematicState(
            frame_idx=self._frame_idx,
            timestamp=timestamp,
            v_raw=v_raw,
            v_ema=v_ema,
            a_t=a_t,
            phase=self._current_phase,
            is_boundary=is_boundary,
            left_wrist=left_wrist,
            right_wrist=right_wrist
        )

        return state, completed_segment

    def _sample_uniform_16(self, buffer: List[np.ndarray]) -> np.ndarray:
        """
        Равномерно сэмплирует ровно 16 кадров из буфера любой длины.
        Возвращает тензор размерности (16, 118).
        """
        n = len(buffer)
        target = self.config.sample_frames  # 16
        if n == 0:
            return np.zeros((target, self.config.feature_dim), dtype=np.float32)

        indices = np.linspace(0, n - 1, target, dtype=int)
        sampled = [buffer[idx] for idx in indices]
        return np.array(sampled, dtype=np.float32)

    def reset(self) -> None:
        """Полный сброс состояния кинематического споттера."""
        self._frame_idx = 0
        self._segment_counter = 0
        self._prev_left_wrist = None
        self._prev_right_wrist = None
        self._prev_v_ema = 0.0
        self._current_phase = KinematicPhase.IDLE
        self._pause_counter = 0
        self._peak_velocity = 0.0
        self._segment_buffer.clear()
