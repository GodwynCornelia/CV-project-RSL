import math

import numpy as np


class GesturePlayback:
    """Presentation time is independent of the 16 inputs used by the classifier."""

    def __init__(self, frame_count=16, duration=2.0, speed=0.5):
        self.speed = speed
        self.configure(frame_count, duration)

    def configure(self, frame_count, duration, *, loop=True, autoplay=True):
        if frame_count < 2 or not math.isfinite(duration) or duration <= 0:
            raise ValueError("Некорректная длительность демонстрации")
        self.frame_count = frame_count
        self.duration = duration
        self.loop = loop
        self.start_hold = 0.5 if loop else 0.0
        self.end_hold = 1.0 if loop else 0.0
        self.position = 0.0
        self._elapsed = -self.start_hold
        self.running = autoplay

    @property
    def movement_duration(self):
        return self.duration / self.speed

    def set_speed(self, speed):
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("Скорость должна быть больше нуля")
        old_duration = self.movement_duration
        if self._elapsed > old_duration:
            self._elapsed = self.duration / speed + self._elapsed - old_duration
        elif self._elapsed >= 0:
            self._elapsed *= self.speed / speed
        self.speed = speed

    def seek(self, position):
        self.position = float(np.clip(position, 0, self.frame_count - 1))
        self._elapsed = self.position / (self.frame_count - 1) * self.movement_duration
        self.running = False

    def play(self):
        if self.position >= self.frame_count - 1:
            self.position = 0.0
            self._elapsed = -self.start_hold
        self.running = True

    def advance(self, seconds):
        if not self.running:
            return
        self._elapsed += max(0.0, seconds)
        duration = self.movement_duration
        if self._elapsed >= duration + self.end_hold:
            if not self.loop:
                self.position = float(self.frame_count - 1)
                self.running = False
                return
            cycle = self.start_hold + duration + self.end_hold
            self._elapsed = (self._elapsed + self.start_hold) % cycle - self.start_hold
        self.position = float(np.clip(self._elapsed / duration, 0, 1) * (self.frame_count - 1))


def interpolate_landmarks(points, valid, position):
    """Interpolate visible joints only; never move a missing hand out of (0, 0)."""
    last = len(points) - 1
    position = float(np.clip(position, 0, last))
    left = int(math.floor(position))
    right = min(left + 1, last)
    fraction = position - left
    nearest = left if fraction < 0.5 else right
    result = points[nearest].copy()
    visible = valid[nearest].copy()
    both = valid[left] & valid[right]
    result[both] = points[left][both] * (1 - fraction) + points[right][both] * fraction
    return result, visible
