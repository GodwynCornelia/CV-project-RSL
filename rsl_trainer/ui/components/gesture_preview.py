from pathlib import Path
import time

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer
from PyQt6.QtGui import QBrush, QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen, QPolygonF, QRadialGradient
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QSlider, QToolButton, QVBoxLayout, QWidget

from rsl_trainer.core.trajectories import read_trajectory
from rsl_trainer.core.playback import GesturePlayback, interpolate_landmarks
from rsl_trainer.core.hand_surface import build_hand_surface, surface_view


def fit_points(points, width, height, padding=18, mirror=False, valid_mask=None):
    """One transform for the whole sequence preserves location and motion."""
    points = np.asarray(points, dtype=np.float32)
    valid = np.isfinite(points).all(axis=-1) & np.any(points != 0, axis=-1)
    if valid_mask is not None:
        valid = np.isfinite(points).all(axis=-1) & valid_mask
    if not valid.any():
        return np.zeros_like(points), valid
    low, high = points[valid].min(axis=0), points[valid].max(axis=0)
    span = np.maximum(high - low, 1e-6)
    scale = min(max(1, width - 2 * padding) / span[0], max(1, height - 2 * padding) / span[1])
    result = (points - (low + high) / 2) * scale
    if mirror:
        result[..., 0] *= -1
    result += [width / 2, height / 2]
    return result, valid


def paint_hand_silhouette(painter, points, color, finger_width):
    """Outlined palm and articulated fingers, retaining their observed geometry.

    Bent fingers are drawn over the palm so a fist does not become an opaque blob.
    This is a 2D rendering of recorded joints, not a reconstruction of hidden depth.
    """
    points = np.asarray(points)
    outline = color.darker(245)
    thickness = max(2.5, float(finger_width))
    fingers = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))

    def stroke(path, width):
        for paint_color, stroke_width in ((outline, width + max(1, thickness * 0.15)), (color, width)):
            pen = QPen(paint_color, stroke_width)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)

    def finger_path(indices):
        path = QPainterPath(QPointF(*points[indices[0]]))
        for index in indices[1:]:
            path.lineTo(QPointF(*points[index]))
        return path

    folded = []
    for finger in fingers:
        base_distance = np.linalg.norm(points[finger[0]] - points[0])
        tip_distance = np.linalg.norm(points[finger[-1]] - points[0])
        if tip_distance < base_distance * 1.25:
            folded.append(finger)
        else:
            stroke(finger_path(finger), thickness)

    across = points[5] - points[17]
    palm = [points[0] + across * 0.38, points[5] + across * 0.07,
            points[9], points[13], points[17] - across * 0.07, points[0] - across * 0.38]
    painter.setPen(QPen(outline, max(1.0, thickness * 0.15)))
    painter.setBrush(color)
    painter.drawPolygon(QPolygonF([QPointF(*point) for point in palm]))
    for finger in folded:
        stroke(finger_path(finger), thickness * 1.05)
        # A short crease keeps the flexed knuckle visible inside a closed hand.
        direction = points[finger[1]] - points[finger[0]]
        perpendicular = np.array([-direction[1], direction[0]])
        perpendicular /= max(np.linalg.norm(perpendicular), 1e-6)
        center = points[finger[1]]
        painter.setPen(QPen(outline, max(0.8, thickness * 0.10)))
        painter.drawLine(QPointF(*(center - perpendicular * thickness * 0.25)),
                         QPointF(*(center + perpendicular * thickness * 0.25)))
    # The thumb is separate from the palm outline, including when it crosses it.
    stroke(finger_path((1, 2, 3, 4)), thickness * 1.15)


_SURFACE_PALETTES = {}


def _paint_volume_front(painter, points, color):
    """Analytic front projection of rounded phalanges avoids per-triangle UI work."""
    points = np.asarray(points)
    axis = points[[5, 9, 13, 17]].mean(axis=0) - points[0]
    length = float(np.linalg.norm(axis))
    if length < 1e-5:
        return
    across = np.array([axis[1], -axis[0]]) / length
    palm_width = max(float(np.ptp(points[[5, 9, 13, 17]] @ across)) * 0.58, length * 0.16)

    def bone(a, b, radius):
        start, end = points[a], points[b]
        delta = end - start
        distance = float(np.linalg.norm(delta))
        if distance < 1e-5:
            return
        transverse = np.array([-delta[1], delta[0]]) / distance * radius
        center = (start + end) / 2
        gradient = QLinearGradient(QPointF(*(center - transverse)), QPointF(*(center + transverse)))
        for stop, shade in ((0, color.darker(190)), (.35, color.lighter(118)), (.65, color), (1, color.darker(145))):
            gradient.setColorAt(stop, shade)
        pen = QPen(QBrush(gradient), radius * 2)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(QPointF(*start), QPointF(*end))
        painter.setPen(Qt.PenStyle.NoPen)
        # Rounded joints are the spherical caps of the same volume.
        for point in (start, end):
            sphere = QRadialGradient(QPointF(*point), radius, QPointF(*(point - radius * .3)))
            sphere.setColorAt(0, color.lighter(120))
            sphere.setColorAt(.6, color)
            sphere.setColorAt(1, color.darker(165))
            painter.setBrush(sphere)
            painter.drawEllipse(QPointF(*point), radius, radius)

    fingers = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
    folded = []
    for finger in fingers:
        if np.linalg.norm(points[finger[-1]] - points[0]) < np.linalg.norm(points[finger[0]] - points[0]) * 1.25:
            folded.append(finger)
        else:
            for a, b in zip(finger, finger[1:]):
                bone(a, b, length * (.072 if a >= 17 else .085))
    painter.save()
    center = points[0] + axis * .56
    painter.translate(float(center[0]), float(center[1]))
    painter.rotate(float(np.degrees(np.arctan2(axis[1], axis[0])) - 90))
    palm_gradient = QRadialGradient(QPointF(-palm_width * .2, -length * .1), max(length, palm_width))
    palm_gradient.setColorAt(0, color.lighter(115))
    palm_gradient.setColorAt(.6, color)
    palm_gradient.setColorAt(1, color.darker(175))
    painter.setPen(QPen(color.darker(170), .6))
    painter.setBrush(palm_gradient)
    painter.drawEllipse(QRectF(-palm_width, -length * .63, palm_width * 2, length * 1.26))
    painter.restore()
    for finger in folded:
        for a, b in zip(finger, finger[1:]):
            bone(a, b, length * (.072 if a >= 17 else .085))
    for a, b in ((0, 1), (1, 2), (2, 3), (3, 4)):
        bone(a, b, length * .105)


def paint_hand_volume(painter, points, color, yaw=0.0, pitch=0.0, viewport=None):
    """Project and shade an actual 3D shell fitted around the recorded joints."""
    if yaw == pitch == 0 and viewport is None and np.shape(points) == (21, 2):
        painter.save()
        _paint_volume_front(painter, points, color)
        painter.restore()
        return
    try:
        surface = build_hand_surface(points)
    except ValueError:
        return
    vertices, normals = surface_view(surface, yaw, pitch)
    if viewport is not None:
        width, height, padding = viewport
        low, high = vertices[:, :2].min(axis=0), vertices[:, :2].max(axis=0)
        span = np.maximum(high - low, 1e-6)
        scale = min((width - padding * 2) / span[0], (height - padding * 2) / span[1])
        vertices[:, :2] = (vertices[:, :2] - (low + high) / 2) * scale + [width / 2, height / 2]
    face_vertices = vertices[surface.faces]
    face_normals = normals[surface.faces].mean(axis=1)
    face_normals /= np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-6)
    visible = face_normals[:, 2] > 0
    face_vertices, face_normals = face_vertices[visible], face_normals[visible]
    light = np.array([-0.35, -0.45, 0.82])
    diffuse = np.clip(face_normals @ light, 0, 1)
    shade = np.clip(0.44 + 0.65 * diffuse + 0.12 * diffuse ** 12, 0, 1.2)
    levels = np.clip(np.round(shade / 1.2 * 47), 0, 47).astype(int)
    key = color.rgb()
    if key not in _SURFACE_PALETTES:
        _SURFACE_PALETTES[key] = [QColor(*(min(255, round(channel * (index / 47 * 1.2)))
                                         for channel in (color.red(), color.green(), color.blue())))
                                  for index in range(48)]
    palette = _SURFACE_PALETTES[key]
    painter.save()
    painter.setPen(Qt.PenStyle.NoPen)
    # Shared triangle edges must not create translucent antialiasing seams.
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
    for index in np.argsort(face_vertices[:, :, 2].mean(axis=1)):
        painter.setBrush(palette[levels[index]])
        painter.drawPolygon(QPolygonF([QPointF(float(x), float(y)) for x, y, _ in face_vertices[index]]))
    painter.restore()


class GestureSilhouettePreview(QWidget):
    """Slow, smoothly interpolated demonstration with pause and phase controls."""

    def __init__(self, parent=None, config=None):
        super().__init__(parent)
        self.setMinimumSize(280, 200)
        self.config = config
        self.trajectory = None
        self.playback = GesturePlayback(speed=config.demonstration_speed if config else 0.5)
        self._recording = False
        self.volume_mode = True
        self._demonstration_duration = 2.0
        self._last_tick = time.monotonic()
        self._geometry_cache = {}
        self.anim_timer = QTimer(self)
        self.anim_timer.setInterval(33)
        self.anim_timer.timeout.connect(self._on_anim_step)
        self._build_controls()

    @property
    def frame_idx(self):
        return int(self.playback.position)

    def _build_controls(self):
        self.controls = QWidget(self)
        self.controls.setStyleSheet("QWidget { background: #102033; } QToolButton, QComboBox { padding: 2px 5px; min-height: 22px; }")
        row = QHBoxLayout(self.controls)
        row.setContentsMargins(8, 2, 8, 2)
        row.setSpacing(5)
        self.previous_button = QToolButton(self.controls)
        self.previous_button.setText("‹")
        self.previous_button.setToolTip("Предыдущая фаза")
        self.previous_button.clicked.connect(lambda: self.step_phase(-1))
        row.addWidget(self.previous_button)
        self.play_button = QToolButton(self.controls)
        self.play_button.clicked.connect(self.toggle_playback)
        row.addWidget(self.play_button)
        self.next_button = QToolButton(self.controls)
        self.next_button.setText("›")
        self.next_button.setToolTip("Следующая фаза")
        self.next_button.clicked.connect(lambda: self.step_phase(1))
        row.addWidget(self.next_button)
        self.speed_combo = QComboBox(self.controls)
        for text, speed in (("0,25×", 0.25), ("0,5×", 0.5), ("0,75×", 0.75), ("1×", 1.0)):
            self.speed_combo.addItem(text, speed)
        self.speed_combo.setCurrentIndex(max(0, self.speed_combo.findData(self.playback.speed)))
        self.speed_combo.setToolTip("Скорость демонстрации; не меняет время записи попытки")
        self.speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        row.addWidget(self.speed_combo)
        self.view_combo = QComboBox(self.controls)
        self.view_combo.addItem("Объём", True)
        self.view_combo.addItem("Контур", False)
        self.view_combo.setToolTip("Объёмная оболочка рассчитана по 2D-точкам. Глубина скрытых суставов приблизительна.")
        self.view_combo.currentIndexChanged.connect(self._on_view_changed)
        row.addWidget(self.view_combo)
        self.phase_slider = QSlider(Qt.Orientation.Horizontal, self.controls)
        self.phase_slider.setRange(0, 15)
        self.phase_slider.setMinimumWidth(40)
        self.phase_slider.setToolTip("Остановить показ на выбранной фазе")
        self.phase_slider.valueChanged.connect(self.seek_phase)
        row.addWidget(self.phase_slider, stretch=1)
        self.phase_label = QLabel(self.controls)
        row.addWidget(self.phase_label)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addStretch()
        layout.addWidget(self.controls)
        self.controls.setFixedHeight(34)
        self.controls.setEnabled(False)
        self._update_controls()

    def set_trajectory(self, trajectory, duration=2.0):
        if isinstance(trajectory, (str, Path)):
            trajectory = read_trajectory(trajectory)
        if trajectory is not None:
            trajectory = np.asarray(trajectory, dtype=np.float32)
            if (trajectory.ndim != 2 or trajectory.shape[1] != 118 or len(trajectory) < 2
                    or not np.isfinite(trajectory).all() or not np.any(trajectory[:, 34:] != 0)):
                trajectory = None
        self.trajectory = trajectory
        self._demonstration_duration = duration
        if self.config is not None:
            self.speed_combo.blockSignals(True)
            self.speed_combo.setCurrentIndex(max(0, self.speed_combo.findData(self.config.demonstration_speed)))
            self.speed_combo.blockSignals(False)
        self._geometry_cache.clear()
        self._recording = False
        self.anim_timer.stop()
        self.playback.configure(len(trajectory) if trajectory is not None else 16, duration,
                                autoplay=trajectory is not None)
        self.controls.setEnabled(trajectory is not None)
        self.phase_slider.setRange(0, self.playback.frame_count - 1)
        if trajectory is not None:
            self.playback.set_speed(self.speed_combo.currentData())
            self._last_tick = time.monotonic()
            self.anim_timer.start()
        self._update_controls()
        self.update()

    def stop_animation(self):
        self.anim_timer.stop()
        self.playback.running = False
        self._update_controls()

    def toggle_playback(self):
        if self.trajectory is None or self._recording:
            return
        if self.playback.running:
            self.stop_animation()
        else:
            self.playback.play()
            self._last_tick = time.monotonic()
            self.anim_timer.start()
            self._update_controls()

    def seek_phase(self, phase):
        if self.trajectory is None or self._recording:
            return
        self.playback.seek(phase)
        self.anim_timer.stop()
        self._update_controls()
        self.update()

    def step_phase(self, direction):
        # At a fractional position, go to the next/previous original observation.
        position = self.playback.position
        phase = np.floor(position) + 1 if direction > 0 else np.ceil(position) - 1
        self.seek_phase(phase)

    def _on_speed_changed(self, index):
        speed = self.speed_combo.currentData()
        self.playback.set_speed(speed)
        if self.config is not None:
            self.config.demonstration_speed = speed
        self._update_controls()

    def _on_view_changed(self, index):
        self.volume_mode = self.view_combo.currentData()
        self.update()

    def begin_recording(self, duration):
        """During an attempt the guide follows recording time, without slow loops."""
        if self.trajectory is None:
            return
        self._recording = True
        self.playback.configure(len(self.trajectory), duration, loop=False)
        self.playback.set_speed(1.0)
        self.controls.setEnabled(False)
        self._last_tick = time.monotonic()
        self.anim_timer.start()
        self._update_controls()
        self.update()

    def finish_recording(self):
        if not self._recording:
            return
        self._recording = False
        self.stop_animation()
        position = self.playback.position
        self.playback.configure(len(self.trajectory), self._demonstration_duration, autoplay=False)
        self.playback.set_speed(self.speed_combo.currentData())
        self.playback.seek(position)
        self.controls.setEnabled(self.trajectory is not None)
        self._update_controls()
        self.update()

    def _update_controls(self):
        self.play_button.setText("Пауза" if self.playback.running else "Пуск")
        self.phase_slider.blockSignals(True)
        self.phase_slider.setValue(round(self.playback.position))
        self.phase_slider.blockSignals(False)
        self.phase_label.setText(f"{self.frame_idx + 1}/{self.playback.frame_count}")

    def _on_anim_step(self):
        now = time.monotonic()
        self.playback.advance(now - self._last_tick)
        self._last_tick = now
        if not self.playback.running:
            self.anim_timer.stop()
        self._update_controls()
        self.update()

    def _drawing_height(self):
        return self.height() - self.controls.height() - 30

    def screen_points(self):
        key = ("body", self.width(), self.height())
        if key not in self._geometry_cache:
            points = self.trajectory.reshape(-1, 59, 2)[:, list(range(11)) + list(range(17, 59))]
            self._geometry_cache[key] = fit_points(points, self.width() * 0.38, self._drawing_height(), padding=16, mirror=True)
        return self._geometry_cache[key]

    def hand_screen_points(self, side):
        key = (side, self.width(), self.height())
        if key not in self._geometry_cache:
            raw = self.trajectory[:, 34 + side * 42:76 + side * 42].reshape(-1, 21, 2)
            valid = np.any(raw != 0, axis=-1)
            local = raw - raw[:, :1]
            panel_width = self.width() * 0.31
            screen, valid = fit_points(local, panel_width, self._drawing_height(),
                                       padding=16, mirror=True, valid_mask=valid)
            screen[..., 0] += self.width() * 0.38 + panel_width * side
            self._geometry_cache[key] = (screen, valid)
        return self._geometry_cache[key]

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#081321"))
        painter.setFont(QFont("Segoe UI", 10))
        if self.trajectory is None:
            painter.setPen(QColor("#94a3b8"))
            area = self.rect().adjusted(10, 0, -10, -self.controls.height())
            painter.drawText(area, Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                             "Эталон движения для этого жеста пока недоступен")
            return
        screen, valid = self.screen_points()
        pts, mask = interpolate_landmarks(screen, valid, self.playback.position)
        pen = QPen(QColor("#536b83"), 7)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        for a, b in ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10)):
            if mask[a] and mask[b]:
                painter.drawLine(QPointF(*pts[a]), QPointF(*pts[b]))
        for offset in (11, 32):
            hand, visible = pts[offset:offset + 21], mask[offset:offset + 21]
            if visible.all():
                width = np.linalg.norm(hand[5] - hand[17]) * 0.20
                paint_hand_silhouette(painter, hand, QColor("#9ce8dd"), max(3, width))
        for side in (0, 1):
            zoom, visible = self.hand_screen_points(side)
            hand, visible = interpolate_landmarks(zoom, visible, self.playback.position)
            if visible.all():
                width = np.linalg.norm(hand[5] - hand[17]) * 0.20
                if self.volume_mode:
                    paint_hand_volume(painter, hand, QColor("#b8f5e9"))
                else:
                    paint_hand_silhouette(painter, hand, QColor("#b8f5e9"), max(3, width))
        painter.setPen(QColor("#cbd5e1"))
        y = self.height() - self.controls.height() - 9
        caption = "Повторяйте движение" if self._recording else "Изучайте движение"
        painter.drawText(14, y, caption)
        painter.drawText(int(self.width() * 0.40), y, "Левая кисть")
        painter.drawText(int(self.width() * 0.72), y, "Правая кисть")
