import time

import numpy as np


class GestureRecording:
    """Keep a whole timed gesture, then sample 16 phases by timestamp.

    Timestamp sampling tolerates changing camera FPS and dropped display frames.
    """

    def __init__(self, duration: float = 2.0):
        self.duration = duration
        self.frames = []
        self.timestamps = []
        self.started_at = None

    def start(self, now=None):
        self.frames.clear()
        self.timestamps.clear()
        self.started_at = time.monotonic() if now is None else now

    def append(self, features, now=None):
        now = time.monotonic() if now is None else now
        if self.started_at is None or np.shape(features) != (118,) or not np.isfinite(features).all():
            return
        self.frames.append(np.asarray(features, dtype=np.float32).copy())
        self.timestamps.append(now)

    def progress(self, now=None):
        if self.started_at is None:
            return 0.0
        now = time.monotonic() if now is None else now
        return min(1.0, max(0.0, (now - self.started_at) / self.duration))

    def sample(self):
        if len(self.frames) < 2:
            raise ValueError("Недостаточно кадров. Проверьте подключение камеры и повторите запись.")
        times = np.asarray(self.timestamps)
        if times[-1] - times[0] < self.duration * 0.6:
            raise ValueError("Запись прервалась. Выполните жест полностью и повторите попытку.")
        targets = np.linspace(times[0], times[-1], 16)
        indices = np.abs(times[:, None] - targets[None, :]).argmin(axis=0)
        sampled = np.asarray(self.frames)[indices]
        visible = np.any(sampled[:, 34:] != 0, axis=1)
        if visible.mean() < 0.5:
            raise ValueError("Кисти плохо видны. Держите руки в кадре во время всего жеста.")
        return sampled
