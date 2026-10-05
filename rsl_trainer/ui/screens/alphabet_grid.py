from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget
)

from rsl_trainer.database.db_manager import DatabaseManager


class AlphabetGridScreen(QWidget):
    """
    Экран А2. Режим «Алфавит» (Сетка букв А-Я по схеме).
    Адаптивная сетка 33 букв русского алфавита с цветовой маркировкой прогресса:
    - Серая: закрыта (locked)
    - Желтая: в процессе (learning)
    - Зеленая: успешно сдана со звездочкой и % совпадения (mastered)
    """

    def __init__(self, navigator, db: DatabaseManager, parent=None):
        super().__init__(parent)
        self.navigator = navigator
        self.db = db

        self._build_ui()

    def _build_ui(self) -> None:
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(32, 20, 32, 24)
        self.main_layout.setSpacing(16)

        # Верхняя панель
        header = QHBoxLayout()
        btn_back = QPushButton("← Назад", self)
        btn_back.setProperty("class", "secondary-btn")
        btn_back.clicked.connect(self.navigator.pop_screen)
        header.addWidget(btn_back)

        title = QLabel("РУССКИЙ ДАКТИЛЬНЫЙ АЛФАВИТ", self)
        title.setStyleSheet("font-size: 18px; font-weight: 800; color: #38bdf8;")
        header.addWidget(title)

        header.addStretch()

        self.progress_summary = QLabel("Освоено: 0 / 33", self)
        self.progress_summary.setStyleSheet("font-weight: bold; color: #10b981; font-size: 14px;")
        header.addWidget(self.progress_summary)

        self.main_layout.addLayout(header)

        # Область прокрутки для сетки букв
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background: transparent; border: none;")

        self.container = QWidget()
        self.container.setStyleSheet("background: transparent;")
        self.grid_layout = QGridLayout(self.container)
        self.grid_layout.setSpacing(12)

        scroll.setWidget(self.container)
        self.main_layout.addWidget(scroll)

    def on_enter(self, **kwargs) -> None:
        """Перерисовка карточек букв из БД при входе на экран."""
        self._refresh_grid()

    def _refresh_grid(self) -> None:
        # Очистка предыдущих виджетов
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        letters_data = self.db.get_dactyl_letters()
        cols = 6  # 6 колонок для комфортной компоновки 33 букв
        mastered_count = 0

        for idx, item in enumerate(letters_data):
            letter = item["letter"]
            status = item["status"]
            score = item["best_score"]

            if status == "mastered":
                mastered_count += 1

            btn = self._create_letter_tile(letter, status, score)
            r = idx // cols
            c = idx % cols
            self.grid_layout.addWidget(btn, r, c)

        self.progress_summary.setText(f"Освоено: {mastered_count} / {len(letters_data)} букв")

    def _create_letter_tile(self, letter: str, status: str, score: float) -> QPushButton:
        btn = QPushButton()
        btn.setProperty("class", f"letter-btn letter-{status}")
        btn.setCursor(Qt.CursorShape.PointingHandCursor)

        # Компоновка содержимого кнопки
        inner = QVBoxLayout(btn)
        inner.setContentsMargins(6, 6, 6, 6)
        inner.setSpacing(2)
        inner.setAlignment(Qt.AlignmentFlag.AlignCenter)

        l_lbl = QLabel(letter)
        l_lbl.setStyleSheet("font-size: 22px; font-weight: 900; background: transparent;")
        l_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inner.addWidget(l_lbl)

        # Индикатор доступности (Иконка + статус для слабовидящих)
        if status == "mastered":
            sub = QLabel(f"⭐ {score*100:.0f}%")
            sub.setStyleSheet("font-size: 10px; color: #34d399; font-weight: bold; background: transparent;")
            sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
            inner.addWidget(sub)
        elif status == "learning":
            sub = QLabel("⏳ Учить")
            sub.setStyleSheet("font-size: 10px; color: #fbbf24; font-weight: bold; background: transparent;")
            sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
            inner.addWidget(sub)
        else:
            sub = QLabel("🔒")
            sub.setStyleSheet("font-size: 10px; color: #64748b; background: transparent;")
            sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
            inner.addWidget(sub)

        # Клик открывает карточку буквы (доступно для learning и mastered)
        btn.clicked.connect(lambda checked, l=letter: self.navigator.push_screen("letter_card", letter=l))
        return btn
