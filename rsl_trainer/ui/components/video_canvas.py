from typing import Optional, Tuple
import cv2
import numpy as np
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout


class VideoCanvas(QFrame):
    """
    Экранный видеохолст.
    - Аппаратное отображение кадров камеры через QPainter
    - Плавное масштабирование с сохранением пропорций
    - Отрисовка статуса удержания и соответствия
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("class", "panel")
        self.setMinimumSize(480, 360)
        self.setStyleSheet("background-color: #050811; border-radius: 12px;")

        self._pixmap: Optional[QPixmap] = None
        self._status_text: str = "📹 Камера не запущена"
        self._hold_progress: float = 0.0  # 0.0 .. 1.0
        self._match_score: float = 0.0    # 0.0 .. 1.0
        self._show_hold_bar: bool = False

    def update_frame(self, frame_bgr: np.ndarray) -> None:
        """Принимает BGR кадр из воркера и конвертирует в QPixmap."""
        h, w, ch = frame_bgr.shape
        bytes_per_line = ch * w
        # BGR -> RGB
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self.update()

    def set_hold_metrics(self, match_score: float, hold_progress: float, show_bar: bool = True) -> None:
        """Обновляет индикаторы точности и времени удержания."""
        self._match_score = match_score
        self._hold_progress = hold_progress
        self._show_hold_bar = show_bar
        self.update()

    def set_placeholder_text(self, text: str) -> None:
        self._status_text = text
        self._pixmap = None
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        if self._pixmap is not None and not self._pixmap.isNull():
            # Масштабируем пиксмап с сохранением пропорций
            scaled = self._pixmap.scaled(
                w, h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            x_off = (w - scaled.width()) // 2
            y_off = (h - scaled.height()) // 2
            painter.drawPixmap(x_off, y_off, scaled)

            # Отрисовка полосы удержания/уверенности внизу
            if self._show_hold_bar:
                self._draw_overlay_hud(painter, w, h)
        else:
            # Заглушка
            painter.setPen(QColor("#64748b"))
            painter.setFont(QFont("Segoe UI", 14, QFont.Weight.DemiBold))
            painter.drawText(QRectF(0, 0, w, h), Qt.AlignmentFlag.AlignCenter, self._status_text)

    def _draw_overlay_hud(self, painter: QPainter, w: int, h: int) -> None:
        """HUD панель со шкалой соответствия и таймером удержания."""
        bar_w = int(w * 0.7)
        bar_h = 20
        bar_x = (w - bar_w) // 2
        bar_y = h - 45

        # Фон шкалы
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(15, 23, 42, 220))
        painter.drawRoundedRect(bar_x - 10, bar_y - 25, bar_w + 20, bar_h + 35, 10, 10)

        # Текст процента
        painter.setPen(QColor("#f8fafc"))
        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        text = f"Совпадение: {self._match_score * 100:.0f}%"
        if self._hold_progress > 0:
            text += f"   ⏱ Удержание: {self._hold_progress * 100:.0f}%"
        painter.drawText(bar_x, bar_y - 6, text)

        # Подложка прогресс-бара
        painter.setBrush(QColor(30, 41, 59))
        painter.drawRoundedRect(bar_x, bar_y, bar_w, bar_h, 6, 6)

        # Заполнение шкалы уверенности (Цветовая градация)
        if self._match_score > 0:
            fill_w = int(bar_w * min(1.0, self._match_score))
            color = QColor("#10b981") if self._match_score >= 0.70 else (
                QColor("#fbbf24") if self._match_score >= 0.40 else QColor("#ef4444")
            )
            painter.setBrush(color)
            painter.drawRoundedRect(bar_x, bar_y, fill_w, bar_h, 6, 6)

        # Золотой индикатор таймера удержания 1.5 сек
        if self._hold_progress > 0:
            hold_w = int(bar_w * self._hold_progress)
            painter.setBrush(QColor(56, 189, 248, 180))  # Cyber blue
            painter.drawRoundedRect(bar_x, bar_y + 4, hold_w, bar_h - 8, 4, 4)
