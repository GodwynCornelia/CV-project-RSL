from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget
)


class InfoScreen(QWidget):
    """
    Экран 2. Справка и калибровка («ИНФО» по схеме).
    Содержит руководство по постановке рук, правила освещения и информацию о проекте.
    """

    def __init__(self, navigator, parent=None):
        super().__init__(parent)
        self.navigator = navigator

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 32)
        layout.setSpacing(16)

        # Верхняя панель
        header = QHBoxLayout()
        btn_back = QPushButton("← Назад", self)
        btn_back.setProperty("class", "secondary-btn")
        btn_back.clicked.connect(self.navigator.pop_screen)
        header.addWidget(btn_back)

        title = QLabel("СПРАВКА И КАЛИБРОВКА КАМЕРЫ", self)
        title.setStyleSheet("font-size: 18px; font-weight: 800; color: #38bdf8;")
        header.addWidget(title)
        header.addStretch()
        layout.addLayout(header)

        # Контентная область с прокруткой
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: transparent; border: none;")

        container = QWidget()
        container.setStyleSheet("background: transparent;")
        c_layout = QVBoxLayout(container)
        c_layout.setSpacing(16)

        # Карточка 1: О Русском Жестовом Языке
        c1 = QFrame()
        c1.setProperty("class", "card")
        c1_l = QVBoxLayout(c1)
        c1_t = QLabel("🤟 Что такое Русский Жестовый Язык (РЖЯ)?")
        c1_t.setStyleSheet("font-size: 16px; font-weight: bold; color: #38bdf8;")
        c1_d = QLabel(
            "РЖЯ — это полноценная естественная лингвистическая система со своей уникальной грамматикой. "
            "В ней используются как статичные формы букв (дактилология), так и пространственные динамические жесты. "
            "Данный тренажер обучен распознавать 999 ключевых жестов и все 33 буквы алфавита."
        )
        c1_d.setWordWrap(True)
        c1_d.setStyleSheet("color: #cbd5e1; line-height: 1.4;")
        c1_l.addWidget(c1_t)
        c1_l.addWidget(c1_d)
        c_layout.addWidget(c1)

        # Карточка 2: Правила работы с камерой
        c2 = QFrame()
        c2.setProperty("class", "card")
        c2_l = QVBoxLayout(c2)
        c2_t = QLabel("📷 Как правильно настроить камеру для максимальной точности:")
        c2_t.setStyleSheet("font-size: 16px; font-weight: bold; color: #10b981;")
        c2_d = QLabel(
            "1. Дистанция: Сядьте на расстоянии 1.0 – 1.5 метра от объектива. Голова, плечи и обе руки должны быть в кадре.\n"
            "2. Освещение: Свет должен падать спереди или сбоку. Избегайте ярких окон позади вас (силуэтного эффекта).\n"
            "3. Для дактиля: Располагайте ладонь внутри пунктирной зоны калибровки («ЗОНА ДЛЯ РУКИ»).\n"
            "4. Для динамических жестов: Делайте короткую паузу (0.3-0.5 сек) перед началом и после окончания движения."
        )
        c2_d.setWordWrap(True)
        c2_d.setStyleSheet("color: #cbd5e1; line-height: 1.5;")
        c2_l.addWidget(c2_t)
        c2_l.addWidget(c2_d)
        c_layout.addWidget(c2)

        # Карточка 3: Авторство и стек
        c3 = QFrame()
        c3.setProperty("class", "card")
        c3_l = QVBoxLayout(c3)
        c3_t = QLabel("🛠 Стек технологий и авторство:")
        c3_t.setStyleSheet("font-size: 16px; font-weight: bold; color: #fbbf24;")
        c3_d = QLabel(
            "• Нейросеть: HierarchicalGestureModel (BiGRU + Spatial Encoders + Mean/Max Pooling)\n"
            "• Компьютерное зрение: Google MediaPipe Tasks API + Ultralytics YOLOv8-Pose\n"
            "• Графика: PyQt6 с оптимизированным CPU конвейером и нулевым лагом очереди\n"
            "• Датасет: Slovo (999 классов)\n"
            "• Разработчик: Федор (@GodwynCornelia)"
        )
        c3_d.setWordWrap(True)
        c3_d.setStyleSheet("color: #cbd5e1; line-height: 1.5;")
        c3_l.addWidget(c3_t)
        c3_l.addWidget(c3_d)
        c_layout.addWidget(c3)

        c_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)
