from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget
)

from rsl_trainer.config import AppConfig


class SettingsScreen(QWidget):
    """
    Экран 3. Настройки приложения («НАСТРОЙКИ» по схеме).
    Управление индексом камеры, порогом чувствительности верификатора и звуком.
    """

    def __init__(self, navigator, config: AppConfig, parent=None):
        super().__init__(parent)
        self.navigator = navigator
        self.config = config

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 32)
        layout.setSpacing(20)

        # Верхняя панель
        header = QHBoxLayout()
        btn_back = QPushButton("← Назад", self)
        btn_back.setProperty("class", "secondary-btn")
        btn_back.clicked.connect(self.navigator.pop_screen)
        header.addWidget(btn_back)

        title = QLabel("ПАРАМЕТРЫ И НАСТРОЙКИ", self)
        title.setStyleSheet("font-size: 18px; font-weight: 800; color: #38bdf8;")
        header.addWidget(title)
        header.addStretch()
        layout.addLayout(header)

        # Контейнер настроек
        card = QFrame(self)
        card.setProperty("class", "panel")
        c_layout = QVBoxLayout(card)
        c_layout.setContentsMargins(24, 24, 24, 24)
        c_layout.setSpacing(24)

        # 1. Выбор камеры
        cam_row = QHBoxLayout()
        cam_lbl = QLabel("Выбор веб-камеры:")
        cam_lbl.setStyleSheet("font-size: 15px; font-weight: bold;")
        self.combo_cam = QComboBox()
        self.combo_cam.addItems(["Камера #0 (Встроенная / Основная)", "Камера #1 (Внешняя)", "Камера #2"])
        self.combo_cam.setCurrentIndex(self.config.camera_id)
        self.combo_cam.setStyleSheet("background-color: #0f172a; padding: 8px 12px; border: 1.5px solid #334155; border-radius: 8px;")
        self.combo_cam.currentIndexChanged.connect(self._on_camera_changed)
        cam_row.addWidget(cam_lbl)
        cam_row.addStretch()
        cam_row.addWidget(self.combo_cam)
        c_layout.addLayout(cam_row)

        # 2. Порог чувствительности валидатора
        thresh_row = QVBoxLayout()
        t_header = QHBoxLayout()
        t_lbl = QLabel("Порог чувствительности распознавания (Confidence):")
        t_lbl.setStyleSheet("font-size: 15px; font-weight: bold;")
        self.thresh_val_lbl = QLabel(f"{int(self.config.dynamic_confidence_threshold * 100)}%")
        self.thresh_val_lbl.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 15px;")
        t_header.addWidget(t_lbl)
        t_header.addStretch()
        t_header.addWidget(self.thresh_val_lbl)
        thresh_row.addLayout(t_header)

        self.slider_thresh = QSlider(Qt.Orientation.Horizontal)
        self.slider_thresh.setRange(20, 60)
        self.slider_thresh.setValue(int(self.config.dynamic_confidence_threshold * 100))
        self.slider_thresh.valueChanged.connect(self._on_thresh_changed)
        thresh_row.addWidget(self.slider_thresh)

        thresh_desc = QLabel("Меньше — легче сдать жест; Больше — строже требования к точности формы руки.")
        thresh_desc.setStyleSheet("color: #64748b; font-size: 12px;")
        thresh_row.addWidget(thresh_desc)
        c_layout.addLayout(thresh_row)

        # 3. Аудио обратная связь
        sound_row = QHBoxLayout()
        self.chk_sound = QCheckBox("Звуковые сигналы успеха / ошибки (для слабовидящих)")
        self.chk_sound.setChecked(self.config.sound_enabled)
        self.chk_sound.setStyleSheet("font-size: 14px;")
        self.chk_sound.toggled.connect(self._on_sound_toggled)
        sound_row.addWidget(self.chk_sound)
        c_layout.addLayout(sound_row)

        # 4. Режим CPU-оптимизации
        opt_box = QFrame()
        opt_box.setStyleSheet("background-color: #0f172a; border-radius: 8px; padding: 12px;")
        opt_l = QVBoxLayout(opt_box)
        opt_title = QLabel("🚀 CPU Оптимизация активирована")
        opt_title.setStyleSheet("color: #10b981; font-weight: bold;")
        opt_desc = QLabel(
            "• Дактиль использует ультра-быстрый трекер только для кисти (55+ FPS).\n"
            "• Динамический режим сжимает разрешение до 320x240 для 4-кратного ускорения YOLO.\n"
            "• Очередь кадров автоматически защищена от задержек (Drop-Frame Buffer)."
        )
        opt_desc.setStyleSheet("color: #94a3b8; font-size: 12px;")
        opt_l.addWidget(opt_title)
        opt_l.addWidget(opt_desc)
        c_layout.addWidget(opt_box)

        layout.addWidget(card)
        layout.addStretch()

    def _on_camera_changed(self, idx: int) -> None:
        self.config.camera_id = idx

    def _on_thresh_changed(self, val: int) -> None:
        self.config.dynamic_confidence_threshold = val / 100.0
        self.thresh_val_lbl.setText(f"{val}%")

    def _on_sound_toggled(self, checked: bool) -> None:
        self.config.sound_enabled = checked
