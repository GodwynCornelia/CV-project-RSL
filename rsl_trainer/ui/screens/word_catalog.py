import json
from pathlib import Path
from typing import Dict, List, Optional
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget
)

from rsl_trainer.config import SKELETONS_DIR
from rsl_trainer.database.db_manager import DatabaseManager, RUSSIAN_ALPHABET


class WordCatalogScreen(QWidget):
    """
    Экраны Б2 и Б3. Словарь РЖЯ (Буквенный рубрикатор + Список слов по схеме).
    - Быстрый алфавитный фильтр (А-Я + Все)
    - Поисковая строка в реальном времени
    - Индикаторы наличия эталонного скелета и статуса изучения
    """

    def __init__(self, navigator, db: DatabaseManager, parent=None):
        super().__init__(parent)
        self.navigator = navigator
        self.db = db

        self.all_words: List[str] = []
        self.cached_skeletons = set()
        self.selected_letter: Optional[str] = None
        self.search_query: str = ""

        self._load_words()
        self._build_ui()

    def _load_words(self) -> None:
        """Загрузка базы слов из label2id.json."""
        label_file = Path(__file__).resolve().parent.parent.parent.parent / "label2id.json"
        if label_file.exists():
            try:
                with open(label_file, "r", encoding="utf-8") as f:
                    data: Dict[str, int] = json.load(f)
                    # Исключаем технический класс no_event и одиночные буквы (они в модуле Дактиль)
                    words = [w for w in data.keys() if w != "no_event" and not (len(w) == 1 and w in RUSSIAN_ALPHABET)]
                    words.sort(key=lambda s: s.lower())
                    self.all_words = words
            except Exception:
                self.all_words = []
        else:
            self.all_words = ["Привет!", "Добро пожаловать!", "вода", "время", "день", "дом", "друг", "любить", "хорошо", "хотеть"]

        # Проверяем, для каких слов есть эталонные npy файлы
        if SKELETONS_DIR.exists():
            for f in SKELETONS_DIR.glob("*.npy"):
                self.cached_skeletons.add(f.stem.lower())

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 20, 28, 24)
        layout.setSpacing(16)

        # Верхняя панель
        header = QHBoxLayout()
        btn_back = QPushButton("← В меню жестов", self)
        btn_back.setProperty("class", "secondary-btn")
        btn_back.clicked.connect(self.navigator.pop_screen)
        header.addWidget(btn_back)

        title = QLabel("СЛОВАРЬ ДИНАМИЧЕСКИХ ЖЕСТОВ", self)
        title.setStyleSheet("font-size: 18px; font-weight: 800; color: #a855f7;")
        header.addWidget(title)
        header.addStretch()

        # Строка поиска
        self.search_input = QLineEdit(self)
        self.search_input.setPlaceholderText("🔍 Поиск по названию жеста...")
        self.search_input.setStyleSheet(
            "background-color: #1e293b; color: #f8fafc; border: 1.5px solid #334155; "
            "border-radius: 10px; padding: 8px 16px; font-size: 14px; min-width: 260px;"
        )
        self.search_input.textChanged.connect(self._on_search_changed)
        header.addWidget(self.search_input)
        layout.addLayout(header)

        # Буквенный рубрикатор (Горизонтальная прокручиваемая лента букв)
        rubric_scroll = QScrollArea(self)
        rubric_scroll.setWidgetResizable(True)
        rubric_scroll.setFixedHeight(56)
        rubric_scroll.setStyleSheet("background: transparent; border: none;")

        rubric_widget = QWidget()
        rubric_layout = QHBoxLayout(rubric_widget)
        rubric_layout.setContentsMargins(0, 4, 0, 4)
        rubric_layout.setSpacing(6)

        self.rubric_group = QButtonGroup(self)
        self.rubric_group.setExclusive(True)

        btn_all = QPushButton("ВСЕ", rubric_widget)
        btn_all.setCheckable(True)
        btn_all.setChecked(True)
        btn_all.setProperty("class", "rubric-btn")
        btn_all.clicked.connect(lambda: self._on_rubric_selected(None))
        self.rubric_group.addButton(btn_all)
        rubric_layout.addWidget(btn_all)

        for char in RUSSIAN_ALPHABET:
            # Проверяем, есть ли слова на эту букву
            has_words = any(w.startswith(char) or w.startswith(char.lower()) for w in self.all_words)
            btn_char = QPushButton(char, rubric_widget)
            btn_char.setCheckable(True)
            btn_char.setEnabled(has_words)
            btn_char.setProperty("class", "rubric-btn")
            btn_char.clicked.connect(lambda checked, c=char: self._on_rubric_selected(c))
            self.rubric_group.addButton(btn_char)
            rubric_layout.addWidget(btn_char)

        rubric_layout.addStretch()
        rubric_scroll.setWidget(rubric_widget)
        layout.addWidget(rubric_scroll)

        # Контейнер списка слов
        self.words_scroll = QScrollArea(self)
        self.words_scroll.setWidgetResizable(True)
        self.words_scroll.setStyleSheet("background: transparent; border: none;")

        self.words_container = QWidget()
        self.words_layout = QVBoxLayout(self.words_container)
        self.words_layout.setContentsMargins(0, 0, 8, 0)
        self.words_layout.setSpacing(10)

        self.words_scroll.setWidget(self.words_container)
        layout.addWidget(self.words_scroll, stretch=1)

        self._refresh_word_list()

    def on_enter(self, **kwargs) -> None:
        self._refresh_word_list()

    def _on_search_changed(self, text: str) -> None:
        self.search_query = text.strip().lower()
        self._refresh_word_list()

    def _on_rubric_selected(self, char: Optional[str]) -> None:
        self.selected_letter = char
        self._refresh_word_list()

    def _refresh_word_list(self) -> None:
        # Очистка предыдущих виджетов
        while self.words_layout.count():
            item = self.words_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        # Фильтрация
        filtered = self.all_words
        if self.selected_letter:
            filtered = [
                w for w in filtered
                if w.startswith(self.selected_letter) or w.startswith(self.selected_letter.lower())
            ]

        if self.search_query:
            filtered = [w for w in filtered if self.search_query in w.lower()]

        if not filtered:
            empty_lbl = QLabel("По заданному запросу слов не найдено.", self.words_container)
            empty_lbl.setStyleSheet("color: #64748b; font-size: 15px; padding: 32px;")
            empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.words_layout.addWidget(empty_lbl)
            return

        # Добавляем элементы списка (ограничим показ первыми 150 для молниеносной отрисовки)
        for word in filtered[:150]:
            card = self._create_word_row(word)
            self.words_layout.addWidget(card)

        if len(filtered) > 150:
            more_lbl = QLabel(f"... и ещё {len(filtered) - 150} слов (уточните поиск)", self.words_container)
            more_lbl.setStyleSheet("color: #64748b; font-size: 13px; padding: 12px;")
            more_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.words_layout.addWidget(more_lbl)

        self.words_layout.addStretch()

    def _create_word_row(self, word: str) -> QWidget:
        row = QFrame(self.words_container)
        row.setStyleSheet(
            "background-color: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 12px 18px;"
        )
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(12, 8, 12, 8)
        row_layout.setSpacing(16)

        # Иконка
        icon = QLabel("🤟", row)
        icon.setStyleSheet("font-size: 24px; background: transparent;")
        row_layout.addWidget(icon)

        # Название жеста
        title_box = QVBoxLayout()
        name_lbl = QLabel(word, row)
        name_lbl.setStyleSheet("font-size: 16px; font-weight: 700; color: #f8fafc; background: transparent;")
        title_box.addWidget(name_lbl)

        # Прогресс из БД
        progress = self.db.get_gesture_progress(word)
        status = progress["status"]
        best_score = progress["best_score"]

        status_text = "Новый жест" if best_score == 0 else f"Сдан ({best_score*100:.0f}%)" if status == "mastered" else f"В процессе ({best_score*100:.0f}%)"
        status_color = "#64748b" if best_score == 0 else "#34d399" if status == "mastered" else "#fbbf24"

        stat_lbl = QLabel(status_text, row)
        stat_lbl.setStyleSheet(f"font-size: 12px; color: {status_color}; background: transparent;")
        title_box.addWidget(stat_lbl)
        row_layout.addLayout(title_box)

        row_layout.addStretch()

        # Бейдж каркаса
        if word.lower() in self.cached_skeletons:
            badge = QLabel("★ Эталонный скелет", row)
            badge.setStyleSheet(
                "background-color: #3b0764; color: #c084fc; border: 1px solid #7e22ce; "
                "border-radius: 6px; padding: 4px 10px; font-size: 11px; font-weight: bold;"
            )
            row_layout.addWidget(badge)

        # Кнопка открытия
        btn_open = QPushButton("Карточка ➔", row)
        btn_open.setProperty("class", "secondary-btn")
        btn_open.clicked.connect(lambda checked, w=word: self._open_word_card(w))
        row_layout.addWidget(btn_open)

        return row

    def _open_word_card(self, word: str) -> None:
        self.navigator.push_screen("gesture_card_view", word=word)
