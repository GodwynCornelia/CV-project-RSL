from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget
)


class DynamicMenuScreen(QWidget):
    """
    Экран Б1. Подменю «Динамические жесты» (по схеме).
    Выбор режима:
    - «Список» (словарь всех жестов и свободная практика)
    - «Предл.» (прохождение уровней Duolingo и сборка фраз)
    """

    def __init__(self, navigator, parent=None):
        super().__init__(parent)
        self.navigator = navigator
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 32)
        layout.setSpacing(24)

        # Верхняя панель
        header = QHBoxLayout()
        btn_back = QPushButton("← В главное меню", self)
        btn_back.setProperty("class", "secondary-btn")
        btn_back.clicked.connect(self.navigator.pop_screen)
        header.addWidget(btn_back)

        title = QLabel("МОДУЛЬ «ДИНАМИЧЕСКИЕ ЖЕСТЫ»", self)
        title.setStyleSheet("font-size: 18px; font-weight: 800; color: #a855f7;")
        header.addWidget(title)
        header.addStretch()
        layout.addLayout(header)

        layout.addStretch()

        # Центральные кнопки выбора режима
        center_layout = QVBoxLayout()
        center_layout.setSpacing(18)

        # Кнопка 1: СПИСОК (Словарь)
        btn_list = QPushButton(self)
        btn_list.setProperty("class", "nav-tile")
        btn_list.clicked.connect(lambda: self.navigator.push_screen("word_catalog"))

        l_layout = QHBoxLayout(btn_list)
        l_icon = QLabel("📚", btn_list)
        l_icon.setStyleSheet("font-size: 38px; background: transparent;")
        l_layout.addWidget(l_icon)
        l_layout.addSpacing(16)

        l_text = QVBoxLayout()
        l_title = QLabel("СЛОВАРЬ ЖЕСТОВ (СПИСОК)", btn_list)
        l_title.setStyleSheet("font-size: 18px; font-weight: 800; color: #f8fafc; background: transparent;")
        l_desc = QLabel("Рубрикатор слов А–Я, поиск по 900+ жестам, эталонные траектории и тренировка", btn_list)
        l_desc.setStyleSheet("font-size: 13px; color: #94a3b8; background: transparent;")
        l_text.addWidget(l_title)
        l_text.addWidget(l_desc)
        l_layout.addLayout(l_text)
        l_layout.addStretch()

        l_arr = QLabel("➔", btn_list)
        l_arr.setStyleSheet("font-size: 22px; color: #a855f7; font-weight: bold; background: transparent;")
        l_layout.addWidget(l_arr)
        center_layout.addWidget(btn_list)

        # Кнопка 2: ПРЕДЛОЖЕНИЯ / УРОВНИ
        btn_levels = QPushButton(self)
        btn_levels.setProperty("class", "nav-tile")
        btn_levels.clicked.connect(lambda: self.navigator.push_screen("sentence_levels"))

        lvl_layout = QHBoxLayout(btn_levels)
        lvl_icon = QLabel("🗺️", btn_levels)
        lvl_icon.setStyleSheet("font-size: 38px; background: transparent;")
        lvl_layout.addWidget(lvl_icon)
        lvl_layout.addSpacing(16)

        lvl_text = QVBoxLayout()
        lvl_title = QLabel("КУРС И ФРАЗЫ (УРОВНИ)", btn_levels)
        lvl_title.setStyleSheet("font-size: 18px; font-weight: 800; color: #f8fafc; background: transparent;")
        lvl_desc = QLabel("Игровое дерево уровней в стиле Duolingo: сборка смысловых фраз жест за жестом", btn_levels)
        lvl_desc.setStyleSheet("font-size: 13px; color: #94a3b8; background: transparent;")
        lvl_text.addWidget(lvl_title)
        lvl_text.addWidget(lvl_desc)
        lvl_layout.addLayout(lvl_text)
        lvl_layout.addStretch()

        lvl_arr = QLabel("➔", btn_levels)
        lvl_arr.setStyleSheet("font-size: 22px; color: #10b981; font-weight: bold; background: transparent;")
        lvl_layout.addWidget(lvl_arr)
        center_layout.addWidget(btn_levels)

        layout.addLayout(center_layout)
        layout.addStretch()
