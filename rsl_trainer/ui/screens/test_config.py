from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget
)


class TestConfigScreen(QWidget):
    """
    Экран А5. Выбор параметров теста (по схеме).
    Выбор объема экзаменационной сессии:
    - 5 букв (Экспресс-разминка)
    - 10 букв (Стандартный тест)
    - 33 буквы (Полный экзамен)
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
        btn_back = QPushButton("← В меню дактиля", self)
        btn_back.setProperty("class", "secondary-btn")
        btn_back.clicked.connect(self.navigator.pop_screen)
        header.addWidget(btn_back)

        title = QLabel("ПАРАМЕТРЫ ЭКЗАМЕНА", self)
        title.setStyleSheet("font-size: 18px; font-weight: 800; color: #38bdf8;")
        header.addWidget(title)
        header.addStretch()
        layout.addLayout(header)

        layout.addStretch()

        # Центральная карточка выбора
        container = QFrame(self)
        container.setStyleSheet(
            "background-color: #1e293b; border: 1.5px solid #334155; border-radius: 16px; padding: 24px;"
        )
        container_layout = QVBoxLayout(container)
        container_layout.setSpacing(20)

        intro_title = QLabel("🎯 Выберите объем проверочной сессии", container)
        intro_title.setStyleSheet("font-size: 20px; font-weight: 800; color: #f8fafc;")
        intro_desc = QLabel(
            "В режиме экзамена у вас будет 5 секунд на воспроизведение каждой случайной буквы.\n"
            "Подсказывающий полупрозрачный каркас будет отключен.",
            container
        )
        intro_desc.setStyleSheet("font-size: 14px; color: #94a3b8; line-height: 1.4;")
        container_layout.addWidget(intro_title)
        container_layout.addWidget(intro_desc)
        container_layout.addSpacing(8)

        # Опция 1: 5 букв
        btn_5 = QPushButton(container)
        btn_5.setProperty("class", "nav-tile")
        btn_5.clicked.connect(lambda: self._start_exam(5))
        self._setup_tile(btn_5, "⚡", "5 БУКВ", "Экспресс-тест (быстрая проверка 5 случайных букв)", "5 мин")
        container_layout.addWidget(btn_5)

        # Опция 2: 10 букв
        btn_10 = QPushButton(container)
        btn_10.setProperty("class", "nav-tile")
        btn_10.clicked.connect(lambda: self._start_exam(10))
        self._setup_tile(btn_10, "⭐", "10 БУКВ", "Стандартная тренировка (рекомендуется для закрепления)", "10 мин")
        container_layout.addWidget(btn_10)

        # Опция 3: 33 буквы
        btn_33 = QPushButton(container)
        btn_33.setProperty("class", "nav-tile")
        btn_33.clicked.connect(lambda: self._start_exam(33))
        self._setup_tile(btn_33, "🏆", "33 БУКВЫ (ВЕСЬ АЛФАВИТ)", "Полный экзамен на знание всех букв русского дактиля", "25 мин")
        container_layout.addWidget(btn_33)

        layout.addWidget(container)
        layout.addStretch()

    def _setup_tile(self, button: QPushButton, icon_char: str, title_text: str, desc_text: str, badge_text: str) -> None:
        btn_layout = QHBoxLayout(button)
        btn_layout.setContentsMargins(16, 12, 16, 12)

        icon_lbl = QLabel(icon_char, button)
        icon_lbl.setStyleSheet("font-size: 32px; background: transparent;")
        btn_layout.addWidget(icon_lbl)
        btn_layout.addSpacing(14)

        text_layout = QVBoxLayout()
        t_lbl = QLabel(title_text, button)
        t_lbl.setStyleSheet("font-size: 16px; font-weight: 800; color: #f8fafc; background: transparent;")
        d_lbl = QLabel(desc_text, button)
        d_lbl.setStyleSheet("font-size: 12px; color: #94a3b8; background: transparent;")
        text_layout.addWidget(t_lbl)
        text_layout.addWidget(d_lbl)
        btn_layout.addLayout(text_layout)
        btn_layout.addStretch()

        badge_lbl = QLabel(badge_text, button)
        badge_lbl.setStyleSheet(
            "background-color: #0f172a; color: #38bdf8; font-size: 12px; font-weight: bold; "
            "padding: 6px 12px; border-radius: 8px; border: 1px solid #0284c7;"
        )
        btn_layout.addWidget(badge_lbl)

    def _start_exam(self, question_count: int) -> None:
        self.navigator.push_screen("exam_camera", question_count=question_count)
