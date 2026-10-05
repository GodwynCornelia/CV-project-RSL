import logging
import time
from typing import Optional
import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from rsl_trainer.config import AppConfig
from rsl_trainer.cv.fast_tracker import FastTracker, TrackerMode

logger = logging.getLogger(__name__)


class CameraWorker(QThread):
    """
    Асинхронный воркер захвата камеры и инференса компьютерного зрения (QThread).
    Полностью изолирует вычисления CV от потока интерфейса GUI.
    Реализует паттерн защиты от очередей (Drop-Frame Queue): обрабатывает только
    самый свежий кадр, гарантируя нулевую задержку отображения.
    """

    # Сигналы в GUI:
    # 1. frame_ready: аннотированный BGR кадр, вектор 118 признаков, координаты кисти (21, 2)
    frame_ready = pyqtSignal(np.ndarray, np.ndarray, object)
    fps_updated = pyqtSignal(float)
    camera_error = pyqtSignal(str)

    def __init__(self, config: Optional[AppConfig] = None):
        super().__init__()
        self.config = config or AppConfig()
        self.tracker = FastTracker(config=self.config)

        self.camera_id = self.config.camera_id
        self._is_running = False
        self._cap: Optional[cv2.VideoCapture] = None

        # Защита от перегрузки очереди
        self._target_fps = self.config.camera_fps
        self._frame_interval = 1.0 / self._target_fps

    def set_mode(self, mode: TrackerMode) -> None:
        """Переключает режим трекинга (DACTYL / DYNAMIC)."""
        self.tracker.set_mode(mode)

    def set_camera_id(self, camera_id: int) -> None:
        """Смена индекса камеры."""
        self.camera_id = camera_id
        if self._is_running:
            self.stop()
            self.start()

    def run(self) -> None:
        self._is_running = True
        try:
            self._cap = cv2.VideoCapture(self.camera_id)
            # Настройка разрешения камеры для легкости обработки
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            if not self._cap.isOpened():
                self.camera_error.emit(f"Не удалось подключиться к камере #{self.camera_id}. Проверьте доступ в системе.")
                self._is_running = False
                return

        except Exception as e:
            self.camera_error.emit(f"Аппаратный сбой инициализации камеры: {e}")
            self._is_running = False
            return

        frame_count = 0
        fps_start = time.perf_counter()

        while self._is_running:
            loop_start = time.perf_counter()

            try:
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    self.camera_error.emit("Потерян сигнал веб-камеры. Попытка переподключения...")
                    time.sleep(0.5)
                    continue

                # Зеркальное отражение для комфортного восприятия пользователя (как в зеркале)
                frame = cv2.flip(frame, 1)

                # 1. Извлечение признаков через FastTracker
                features_118, primary_hand, body_pts = self.tracker.extract_features(frame)

                # 2. Отрисовка скелета на кадре
                annotated_frame = self.tracker.draw_skeleton(
                    frame,
                    primary_hand=primary_hand,
                    body_pts=body_pts
                )

                # 3. Отправка в GUI поток
                self.frame_ready.emit(annotated_frame, features_118, primary_hand)

                # Подсчет FPS
                frame_count += 1
                if frame_count % 30 == 0:
                    now = time.perf_counter()
                    real_fps = frame_count / (now - fps_start)
                    self.fps_updated.emit(round(real_fps, 1))
                    frame_count = 0
                    fps_start = now

                # Контроль частоты кадров (sleep только если цикл выполнился быстрее интервала)
                elapsed = time.perf_counter() - loop_start
                sleep_time = self._frame_interval - elapsed
                if sleep_time > 0.001:
                    time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Ошибка в цикле захвата камеры: {e}")
                self.camera_error.emit(f"Ошибка обработки кадра: {e}")
                time.sleep(0.1)

        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def stop(self) -> None:
        """Безопасная остановка потока."""
        self._is_running = False
        self.wait(1200)
