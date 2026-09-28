import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPainter, QPixmap
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout


class VideoWidget(QFrame):
    """
    Экранный холст видео (Video Canvas).
    Отображает видеопоток с наложением скелета тела и кистей рук.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("class", "panel")
        self.setMinimumSize(640, 480)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet("background-color: #050811; border-radius: 8px;")
        layout.addWidget(self.image_label)

        self._show_placeholder()

    def _show_placeholder(self) -> None:
        self.image_label.setText(
            "📹 Камера готова к запуску\n"
            "Нажмите «Запустить камеру» или используйте «Демо-поток»"
        )
        self.image_label.setStyleSheet("color: #64748b; font-size: 14px; background-color: #050811; border-radius: 8px;")

    def update_frame(self, frame_bgr: np.ndarray) -> None:
        """Принимает BGR numpy кадр, конвертирует в QPixmap и выводит на холст."""
        h, w, ch = frame_bgr.shape
        bytes_per_line = ch * w
        # Конвертация BGR -> RGB
        rgb_data = frame_bgr[:, :, ::-1].copy()
        q_img = QImage(rgb_data.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        # Масштабирование под размер виджета с сохранением пропорций
        scaled_pixmap = QPixmap.fromImage(q_img).scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
