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


class ExamCardWidget(QFrame):
    """
    Интерактивная карточка выбора сессии экзамена фиксированной высоты (88 px).
    Изолированные стили по ID #ExamCard предотвращают наслоение границ.
    """

    def __init__(self, icon_char: str, title_text: str, desc_text: str, badge_text: str, on_click, parent=None):
        super().__init__(parent)
        self.setObjectName("ExamCard")
        self.setFixedHeight(88)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.on_click = on_click

        self.setStyleSheet("""
            QFrame#ExamCard {
                background-color: #1a2332;
                border: 1px solid #2a3b50;
                border-radius: 12px;
            }
            QFrame#ExamCard:hover {
                border: 1.5px solid #00b4d8;
                background-color: #1e2c3f;
            }
            QFrame#ExamCard QFrame, QFrame#ExamCard QLabel {
                border: none;
                background: transparent;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 12, 18, 12)
        layout.setSpacing(16)

        # 1. Слева: Иконка (64 px)
        icon_lbl = QLabel(icon_char, self)
        icon_lbl.setFixedWidth(64)
        icon_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_lbl.setStyleSheet("font-size: 32px;")
        layout.addWidget(icon_lbl)

        # 2. По центру: Заголовок и описание (QVBoxLayout)
        info_layout = QVBoxLayout()
        info_layout.setContentsMargins(0, 0, 0, 0)
        info_layout.setSpacing(4)
        info_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter)

        t_lbl = QLabel(title_text, self)
        t_lbl.setStyleSheet("font-size: 16px; font-weight: 800; color: #ffffff;")
        info_layout.addWidget(t_lbl)

        d_lbl = QLabel(desc_text, self)
        d_lbl.setWordWrap(True)
        d_lbl.setStyleSheet("font-size: 12px; color: #8fa3bf;")
        info_layout.addWidget(d_lbl)

        layout.addLayout(info_layout, stretch=1)

        # 3. Справа: Бейдж времени (80 px)
        badge_lbl = QLabel(badge_text, self)
        badge_lbl.setFixedWidth(80)
        badge_lbl.setFixedHeight(34)
        badge_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge_lbl.setStyleSheet("""
            background-color: #0b1a2d;
            color: #00b4d8;
            font-size: 12px;
            font-weight: bold;
            border-radius: 8px;
            border: 1px solid #0077b6;
        """)
        layout.addWidget(badge_lbl)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.on_click()
        super().mousePressEvent(event)


class TestConfigScreen(QWidget):
    """
    Экран А5. Выбор параметров теста (по схеме).
    - Обернут в QScrollArea с автоматическим ресайзом
    - Изолированные карточки ExamCardWidget высотой 88 px
    - Безопасная навигация в Экран А6 (Камера-экзамен)
    """

    def __init__(self, navigator, parent=None):
        super().__init__(parent)
        self.navigator = navigator
        self._build_ui()

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(32, 24, 32, 24)
        main_layout.setSpacing(16)

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
        main_layout.addLayout(header)

        # QScrollArea для надежного скроллинга на любых разрешениях
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background: transparent; border: none;")

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(0, 12, 12, 12)
        scroll_layout.setSpacing(16)
        scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Вводный блок
        intro_box = QFrame(scroll_widget)
        intro_box.setStyleSheet(
            "background-color: #131b2e; border: 1px solid #1e293b; border-radius: 14px; padding: 18px 24px;"
        )
        intro_layout = QVBoxLayout(intro_box)
        intro_layout.setSpacing(6)

        intro_title = QLabel("🎯 Выберите объем проверочной сессии", intro_box)
        intro_title.setStyleSheet("font-size: 18px; font-weight: 800; color: #f8fafc;")
        intro_desc = QLabel(
            "В режиме экзамена у вас будет 5 секунд на воспроизведение каждой случайной буквы.\n"
            "Подсказывающий полупрозрачный каркас будет отключен.",
            intro_box
        )
        intro_desc.setStyleSheet("font-size: 13px; color: #94a3b8; line-height: 1.4;")
        intro_layout.addWidget(intro_title)
        intro_layout.addWidget(intro_desc)
        scroll_layout.addWidget(intro_box)

        # Карточка 1: 5 букв
        card_5 = ExamCardWidget(
            icon_char="⚡",
            title_text="5 БУКВ",
            desc_text="Экспресс-тест (быстрая проверка 5 случайных букв)",
            badge_text="5 мин",
            on_click=lambda: self._start_exam(5),
            parent=scroll_widget
        )
        scroll_layout.addWidget(card_5)

        # Карточка 2: 10 букв
        card_10 = ExamCardWidget(
            icon_char="⭐",
            title_text="10 БУКВ",
            desc_text="Стандартная тренировка (рекомендуется для закрепления)",
            badge_text="10 мин",
            on_click=lambda: self._start_exam(10),
            parent=scroll_widget
        )
        scroll_layout.addWidget(card_10)

        # Карточка 3: 33 буквы
        card_33 = ExamCardWidget(
            icon_char="🏆",
            title_text="33 БУКВЫ (ВЕСЬ АЛФАВИТ)",
            desc_text="Полный экзамен на знание всех букв русского дактиля",
            badge_text="25 мин",
            on_click=lambda: self._start_exam(33),
            parent=scroll_widget
        )
        scroll_layout.addWidget(card_33)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        main_layout.addWidget(scroll, stretch=1)

    def _start_exam(self, question_count: int) -> None:
        self.navigator.push_screen("exam_camera", question_count=question_count)
