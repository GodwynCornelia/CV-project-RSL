from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget
)


class DactylMenuScreen(QWidget):
    """
    Экран А1. Подменю «Дактиль» (по схеме).
    Выбор между пошаговым обучением («Алфавит») и экзаменом («Тест»).
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

        title = QLabel("МОДУЛЬ «ДАКТИЛЬ»", self)
        title.setStyleSheet("font-size: 18px; font-weight: 800; color: #38bdf8;")
        header.addWidget(title)
        header.addStretch()
        layout.addLayout(header)

        layout.addStretch()

        # Центральные кнопки выбора режима
        center_layout = QVBoxLayout()
        center_layout.setSpacing(18)

        # Кнопка 1: АЛФАВИТ
        btn_alphabet = QPushButton(self)
        btn_alphabet.setProperty("class", "nav-tile")
        btn_alphabet.clicked.connect(lambda: self.navigator.push_screen("alphabet_grid"))

        ab_l = QHBoxLayout(btn_alphabet)
        ab_icon = QLabel("📖", btn_alphabet)
        ab_icon.setStyleSheet("font-size: 38px; background: transparent;")
        ab_l.addWidget(ab_icon)
        ab_l.addSpacing(16)

        ab_text = QVBoxLayout()
        ab_title = QLabel("АЛФАВИТ (ОБУЧЕНИЕ)", btn_alphabet)
        ab_title.setStyleSheet("font-size: 18px; font-weight: 800; color: #f8fafc; background: transparent;")
        ab_desc = QLabel("Сетка 33 букв, эталонные позы пальцев и интерактивная камера с полупрозрачным каркасом", btn_alphabet)
        ab_desc.setStyleSheet("font-size: 13px; color: #94a3b8; background: transparent;")
        ab_text.addWidget(ab_title)
        ab_text.addWidget(ab_desc)
        ab_l.addLayout(ab_text)
        ab_l.addStretch()

        ab_arr = QLabel("➔", btn_alphabet)
        ab_arr.setStyleSheet("font-size: 22px; color: #38bdf8; font-weight: bold; background: transparent;")
        ab_l.addWidget(ab_arr)
        center_layout.addWidget(btn_alphabet)

        # Кнопка 2: ТЕСТ
        btn_test = QPushButton(self)
        btn_test.setProperty("class", "nav-tile")
        btn_test.clicked.connect(lambda: self.navigator.push_screen("test_config"))

        t_l = QHBoxLayout(btn_test)
        t_icon = QLabel("⏱", btn_test)
        t_icon.setStyleSheet("font-size: 38px; background: transparent;")
        t_l.addWidget(t_icon)
        t_l.addSpacing(16)

        t_text = QVBoxLayout()
        t_title = QLabel("ТЕСТ (ЭКЗАМЕН НА ВРЕМЯ)", btn_test)
        t_title.setStyleSheet("font-size: 18px; font-weight: 800; color: #f8fafc; background: transparent;")
        t_desc = QLabel("Проверка знаний без подсказок: случайные буквы, таймер 5 секунд и итоговый балл", btn_test)
        t_desc.setStyleSheet("font-size: 13px; color: #94a3b8; background: transparent;")
        t_text.addWidget(t_title)
        t_text.addWidget(t_desc)
        t_l.addLayout(t_text)
        t_l.addStretch()

        t_arr = QLabel("➔", btn_test)
        t_arr.setStyleSheet("font-size: 22px; color: #10b981; font-weight: bold; background: transparent;")
        t_l.addWidget(t_arr)
        center_layout.addWidget(btn_test)

        layout.addLayout(center_layout)
        layout.addStretch()
