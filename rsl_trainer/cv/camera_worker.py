import logging
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QThread, QTimer, pyqtSignal

from rsl_trainer.config import AppConfig
from rsl_trainer.cv.fast_tracker import FastTracker, TrackerMode

logger = logging.getLogger(__name__)


class CameraWorker(QThread):
    """Capture and CV run in this thread; the GUI polls one latest packet.

    Features and primary_hand retain original camera coordinates, as in training.
    Only the annotated preview is mirrored for the user.
    """

    frame_ready = pyqtSignal(np.ndarray, np.ndarray, object)
    fps_updated = pyqtSignal(float)
    camera_error = pyqtSignal(str)

    def __init__(self, config: Optional[AppConfig] = None):
        super().__init__()
        self.config = config or AppConfig()
        self.camera_id = self.config.camera_id
        self.tracker = None
        self._mode = TrackerMode.DACTYL
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._pending_frame = None
        self._cap = None
        self._delivery_timer = QTimer(self)
        self._delivery_timer.setInterval(max(1, round(1000 / self.config.camera_fps)))
        self._delivery_timer.timeout.connect(self._deliver_latest)

    def start(self, priority=QThread.Priority.InheritPriority) -> None:
        if self.isRunning():
            return
        self._stop_event.clear()
        self._delivery_timer.start()
        super().start(priority)

    def set_mode(self, mode: TrackerMode) -> None:
        self._mode = mode

    def set_camera_id(self, camera_id: int) -> None:
        was_running = self.isRunning()
        if was_running and not self.stop():
            self.camera_error.emit("Не удалось остановить предыдущую камеру. Повторите попытку.")
            return
        self.camera_id = camera_id
        self.config.camera_id = camera_id
        if was_running:
            self.start()

    def _publish_frame(self, frame, features, hand) -> None:
        with self._lock:
            self._pending_frame = (frame, features, hand)

    def _deliver_latest(self) -> None:
        with self._lock:
            packet, self._pending_frame = self._pending_frame, None
        if packet is not None:
            self.frame_ready.emit(*packet)

    def run(self) -> None:
        try:
            backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
            self._cap = cv2.VideoCapture(self.camera_id, backend)
            if not self._cap.isOpened():
                raise RuntimeError(f"Веб-камера #{self.camera_id} недоступна")
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if self._stop_event.is_set():
                return
            self.tracker = FastTracker(config=self.config)
            consecutive_fails = 0
            frame_count = 0
            fps_start = time.monotonic()
            while not self._stop_event.is_set():
                loop_start = time.monotonic()
                ret, frame = self._cap.read()
                if not ret or frame is None:
                    consecutive_fails += 1
                    if consecutive_fails >= 10:
                        raise RuntimeError("Камера перестала передавать кадры")
                    self._stop_event.wait(0.05)
                    continue
                consecutive_fails = 0
                self.tracker.set_mode(self._mode)
                # Training clips are unmirrored: mirror only the displayed image.
                features, hand, body = self.tracker.extract_features(frame)
                annotated = self.tracker.draw_skeleton(frame, primary_hand=hand, body_pts=body)
                self._publish_frame(cv2.flip(annotated, 1), features, hand)
                frame_count += 1
                if frame_count == 30:
                    now = time.monotonic()
                    self.fps_updated.emit(frame_count / max(now - fps_start, 1e-6))
                    frame_count, fps_start = 0, now
                interval = 1 / self.config.camera_fps
                self._stop_event.wait(max(0, interval - (time.monotonic() - loop_start)))
        except Exception as exc:
            logger.exception("Camera capture failed")
            if not self._stop_event.is_set():
                self.camera_error.emit(str(exc))
        finally:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            if self.tracker is not None:
                self.tracker.close()
                self.tracker = None
            with self._lock:
                self._pending_frame = None

    def stop(self, timeout_ms: int = 3000) -> bool:
        self._delivery_timer.stop()
        self._stop_event.set()
        stopped = self.wait(timeout_ms)
        with self._lock:
            self._pending_frame = None
        return stopped
