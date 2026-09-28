from collections import deque
from typing import Optional
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout

from src.models import KinematicPhase, KinematicState


class KinematicsOscillographWidget(QFrame):
    """
    Патентоспособный осциллограф кинематики кистей рук в реальном времени.
    Отображает:
    - График дифференциальной скорости движения кистей (EMA).
    - Горизонтальную красную пунктирную линию порога отсечки (Threshold_stop).
    - Индикатор активной кинематической фазы (IDLE, PREPARATION, STROKE, RETRACTION, TRIGGER).
    - Мгновенные значения скорости и ускорения.
    """

    def __init__(self, buffer_size: int = 150, parent=None):
        super().__init__(parent)
        self.setProperty("class", "panel")
        self.setMinimumHeight(170)

        self.buffer_size = buffer_size
        self.velocity_history = deque([0.0] * buffer_size, maxlen=buffer_size)
        self.threshold_stop: float = 7.0
        self.threshold_active: float = 16.0

        self.current_state: Optional[KinematicState] = None

    def update_kinematics(self, state: KinematicState) -> None:
        """Обновление точки графика на каждом кадре."""
        self.current_state = state
        self.velocity_history.append(state.v_ema)
        self.update()  # Вызывает перерисовку paintEvent

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        padding_top = 35
        padding_bottom = 25
        padding_left = 45
        padding_right = 20

        plot_w = w - padding_left - padding_right
        plot_h = h - padding_top - padding_bottom

        # 1. Фоновая подложка графика
        painter.fillRect(padding_left, padding_top, plot_w, plot_h, QColor("#090e1a"))

        # 2. Координатная сетка
        grid_pen = QPen(QColor("#1e293b"), 1, Qt.PenStyle.DashLine)
        painter.setPen(grid_pen)
        for y_step in [0.25, 0.5, 0.75]:
            y_pos = padding_top + plot_h * y_step
            painter.drawLine(padding_left, int(y_pos), padding_left + plot_w, int(y_pos))

        # Максимальная шкала скорости (автомасштабирование)
        max_val = max(35.0, max(self.velocity_history) * 1.25)

        # 3. Красная пунктирная линия порога отсечки (Threshold Stop)
        norm_stop_y = padding_top + plot_h * (1.0 - min(1.0, self.threshold_stop / max_val))
        thresh_pen = QPen(QColor("#f43f5e"), 1.8, Qt.PenStyle.DashDotLine)
        painter.setPen(thresh_pen)
        painter.drawLine(padding_left, int(norm_stop_y), padding_left + plot_w, int(norm_stop_y))

        # Текст порога
        painter.setPen(QColor("#f43f5e"))
        painter.setFont(QFont("Segoe UI", 8, QFont.Weight.Bold))
        painter.drawText(padding_left + 8, int(norm_stop_y) - 4, f"Порог отсечки: {self.threshold_stop:.1f} px/f")

        # 4. Построение кривой скорости кистей рук
        if len(self.velocity_history) > 1:
            points = []
            dx = plot_w / (self.buffer_size - 1)
            for idx, vel in enumerate(self.velocity_history):
                x = padding_left + idx * dx
                norm_y = padding_top + plot_h * (1.0 - min(1.0, vel / max_val))
                points.append(QPointF(x, norm_y))

            # Создание пути для заливки градиентом
            fill_path = QPainterPath()
            fill_path.moveTo(padding_left, padding_top + plot_h)
            for pt in points:
                fill_path.lineTo(pt)
            fill_path.lineTo(padding_left + plot_w, padding_top + plot_h)
            fill_path.closeSubpath()

            gradient = QLinearGradient(0, padding_top, 0, padding_top + plot_h)
            gradient.setColorAt(0.0, QColor(56, 189, 248, 120))   # Cyber blue semi-transparent
            gradient.setColorAt(1.0, QColor(56, 189, 248, 10))
            painter.fillPath(fill_path, gradient)

            # Отрисовка линии кривой
            curve_pen = QPen(QColor("#38bdf8"), 2.2)
            painter.setPen(curve_pen)
            for i in range(len(points) - 1):
                painter.drawLine(points[i], points[i + 1])

        # 5. Заголовок и текущие числовые метрики
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(padding_left, 22, "КИСЛОГРАФМА КИНЕМАТИКИ ЗАПЯСТИЙ (EMA СКОРОСТЬ)")

        if self.current_state is not None:
            v_val = self.current_state.v_ema
            a_val = self.current_state.a_t
            phase = self.current_state.phase

            # Цвет индикатора фазы
            phase_colors = {
                KinematicPhase.IDLE: QColor("#64748b"),
                KinematicPhase.PREPARATION: QColor("#38bdf8"),
                KinematicPhase.STROKE: QColor("#10b981"),
                KinematicPhase.RETRACTION: QColor("#fbbf24"),
                KinematicPhase.TRIGGER: QColor("#f43f5e"),
            }
            p_color = phase_colors.get(phase, QColor("#64748b"))

            # Бейдж фазы в верхнем правом углу
            badge_text = f"ФАЗА: {phase.value}"
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.setPen(p_color)
            painter.drawText(w - padding_right - 180, 22, badge_text)

            # Метрика текущей скорости
            val_text = f"V: {v_val:.1f} px/f   A: {a_val:+.1f}"
            painter.setFont(QFont("Segoe UI", 9))
            painter.setPen(QColor("#cbd5e1"))
            painter.drawText(w - padding_right - 340, 22, val_text)
