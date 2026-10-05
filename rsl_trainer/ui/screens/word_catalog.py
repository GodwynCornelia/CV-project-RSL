import json
from pathlib import Path
from typing import Dict, List, Optional, Set

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

# Определение тематических категорий по спецификации Patch v1.2
CATEGORY_SPECS = [
    ("all", "Все", "🌐", "Жест"),
    ("top_50", "Топ-50 базовых", "⭐", "Базовый"),
    ("greetings", "Знакомство и вежливость", "🤝", "Этикет"),
    ("family", "Семья и люди", "👨‍👩‍👦", "Семья"),
    ("home", "Дом и быт", "🏠", "Дом"),
    ("food", "Еда и напитки", "🍎", "Еда"),
    ("city_transport", "Город и транспорт", "🚗", "Город"),
    ("time_numbers", "Время и числа", "⏰", "Время"),
    ("verbs", "Действия / Глаголы", "⚡", "Глагол"),
]

TAB_ACTIVE_STYLE = """
QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7c3aed, stop:1 #9333ea);
    color: #ffffff;
    font-size: 13px;
    font-weight: 700;
    border: 1.5px solid #a855f7;
    border-radius: 16px;
    padding: 6px 16px;
    min-height: 32px;
}
"""

TAB_INACTIVE_STYLE = """
QPushButton {
    background-color: #1e293b;
    color: #94a3b8;
    font-size: 13px;
    font-weight: 600;
    border: 1.5px solid #334155;
    border-radius: 16px;
    padding: 6px 16px;
    min-height: 32px;
}
QPushButton:hover {
    background-color: #334155;
    border-color: #475569;
    color: #f8fafc;
}
"""

RUBRIC_ACTIVE_STYLE = """
QPushButton {
    background-color: #0284c7;
    color: #ffffff;
    font-weight: 800;
    font-size: 13px;
    border: 1.5px solid #38bdf8;
    border-radius: 6px;
    min-width: 32px;
    min-height: 28px;
}
"""

RUBRIC_INACTIVE_STYLE = """
QPushButton {
    background-color: #1e293b;
    color: #cbd5e1;
    font-weight: 700;
    font-size: 13px;
    border: 1px solid #334155;
    border-radius: 6px;
    min-width: 32px;
    min-height: 28px;
}
QPushButton:hover {
    background-color: #334155;
    color: #ffffff;
}
QPushButton:disabled {
    background-color: #0f172a;
    color: #475569;
    border-color: #1e293b;
}
"""


class GestureCardWidget(QFrame):
    """Аккуратная карточка жеста фиксированного размера с бейджем категории и статусом."""

    def __init__(
        self,
        word: str,
        category_name: str,
        has_skeleton: bool,
        progress: dict,
        on_click,
        parent=None
    ):
        super().__init__(parent)
        self.word = word
        self.on_click = on_click
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(115)
        self.setMinimumWidth(220)

        self.setObjectName("gestureCard")
        self.setStyleSheet("""
            QFrame#gestureCard {
                background-color: #182238;
                border: 1.5px solid #29354f;
                border-radius: 12px;
            }
            QFrame#gestureCard:hover {
                background-color: #1e2b47;
                border: 1.5px solid #a855f7;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(6)

        # Верхняя строка: бейдж категории + метка скелета
        top_row = QHBoxLayout()
        top_row.setSpacing(6)

        cat_badge = QLabel(category_name, self)
        cat_badge.setStyleSheet(
            "background-color: #2e1065; color: #c084fc; border: 1px solid #581c87; "
            "border-radius: 6px; padding: 2px 8px; font-size: 11px; font-weight: bold;"
        )
        top_row.addWidget(cat_badge)

        if has_skeleton:
            skel_badge = QLabel("★ 3D", self)
            skel_badge.setStyleSheet(
                "background-color: #064e3b; color: #34d399; border: 1px solid #059669; "
                "border-radius: 6px; padding: 2px 6px; font-size: 10px; font-weight: bold;"
            )
            top_row.addWidget(skel_badge)

        top_row.addStretch()
        layout.addLayout(top_row)

        # Название жеста
        name_lbl = QLabel(word, self)
        name_lbl.setStyleSheet(
            "font-size: 15px; font-weight: 700; color: #f8fafc; background: transparent;"
        )
        name_lbl.setWordWrap(True)
        layout.addWidget(name_lbl)

        layout.addStretch()

        # Нижняя строка: статус освоения + переход
        bottom_row = QHBoxLayout()
        bottom_row.setContentsMargins(0, 0, 0, 0)

        status = progress.get("status", "locked")
        best_score = progress.get("best_score", 0.0)

        if status == "mastered" or best_score >= 0.8:
            status_text = f"✓ Освоен ({int(best_score * 100)}%)"
            status_color = "#34d399"
        elif best_score > 0:
            status_text = f"⏳ В процессе ({int(best_score * 100)}%)"
            status_color = "#fbbf24"
        else:
            status_text = "○ Не начат"
            status_color = "#64748b"

        stat_lbl = QLabel(status_text, self)
        stat_lbl.setStyleSheet(f"font-size: 11px; font-weight: 600; color: {status_color}; background: transparent;")
        bottom_row.addWidget(stat_lbl)

        bottom_row.addStretch()

        arrow_lbl = QLabel("Практика ➔", self)
        arrow_lbl.setStyleSheet("font-size: 11px; font-weight: bold; color: #a855f7; background: transparent;")
        bottom_row.addWidget(arrow_lbl)

        layout.addLayout(bottom_row)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.on_click(self.word)
        super().mousePressEvent(event)


class WordCatalogScreen(QWidget):
    """
    Экран «Каталог динамических жестов» (Patch v1.2):
    1. Горизонтальная панель тематических табов-пилюль (Все, Топ-50, Еда, Семья...)
    2. Быстрый поиск на лету (Live Search с мгновенной фильтрацией textChanged)
    3. Алфавитный рубрикатор (А-Я)
    4. Сетка карточек фиксированного размера с бейджами, статусом освоения и переходом.
    """

    CHUNK_SIZE = 100  # Порция карточек для мгновенной 60 FPS отрисовки

    def __init__(self, navigator, db: DatabaseManager, parent=None):
        super().__init__(parent)
        self.navigator = navigator
        self.db = db

        self.all_words: List[str] = []
        self.cached_skeletons: Set[str] = set()

        self.categories_map: Dict[str, List[str]] = {}
        self.word_badge_map: Dict[str, str] = {}

        self.selected_category: str = "all"
        self.selected_letter: Optional[str] = None
        self.search_query: str = ""

        self.displayed_count: int = self.CHUNK_SIZE
        self.filtered_words: List[str] = []

        self.category_buttons: Dict[str, QPushButton] = {}
        self.rubric_buttons: Dict[str, QPushButton] = {}

        self._load_data()
        self._build_ui()

    def _load_data(self) -> None:
        """Загрузка базы слов, скелетов и категорий."""
        # 1. Загрузка слов из label2id.json
        label_file = Path(__file__).resolve().parent.parent.parent.parent / "label2id.json"
        if label_file.exists():
            try:
                with open(label_file, "r", encoding="utf-8") as f:
                    data: Dict[str, int] = json.load(f)
                    words = [
                        w for w in data.keys()
                        if w != "no_event" and not (len(w) == 1 and w in RUSSIAN_ALPHABET)
                    ]
                    words.sort(key=lambda s: s.lower())
                    self.all_words = words
            except Exception:
                self.all_words = []
        else:
            self.all_words = [
                "Привет!", "Добро пожаловать!", "вода", "время", "день",
                "дом", "друг", "любить", "хорошо", "хотеть"
            ]

        # 2. Кэш скелетов
        if SKELETONS_DIR.exists():
            for f in SKELETONS_DIR.glob("*.npy"):
                self.cached_skeletons.add(f.stem.lower())

        # 3. Загрузка тематических категорий
        cat_file = Path(__file__).resolve().parent.parent.parent / "data" / "gesture_categories.json"
        if cat_file.exists():
            try:
                with open(cat_file, "r", encoding="utf-8") as f:
                    self.categories_map = json.load(f)
            except Exception:
                self.categories_map = {}

        # 4. Построение обратного маппинга слова в бейдж категории
        badge_priority = [
            ("food", "Еда"),
            ("family", "Семья"),
            ("home", "Дом"),
            ("city_transport", "Город"),
            ("time_numbers", "Время"),
            ("verbs", "Глагол"),
            ("greetings", "Этикет"),
            ("top_50", "Базовый"),
        ]
        for cat_id, badge_label in badge_priority:
            for w in self.categories_map.get(cat_id, []):
                if w not in self.word_badge_map:
                    self.word_badge_map[w] = badge_label

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 18, 28, 20)
        layout.setSpacing(12)

        # 1. Верхняя строка заголовка и поиска
        header = QHBoxLayout()
        header.setSpacing(16)

        btn_back = QPushButton("← В меню жестов", self)
        btn_back.setProperty("class", "secondary-btn")
        btn_back.clicked.connect(self.navigator.pop_screen)
        header.addWidget(btn_back)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title = QLabel("КАТАЛОГ ДИНАМИЧЕСКИХ ЖЕСТОВ", self)
        title.setStyleSheet("font-size: 18px; font-weight: 800; color: #a855f7;")
        title_box.addWidget(title)

        self.subtitle = QLabel(f"Всего доступно {len(self.all_words)} классов РЖЯ", self)
        self.subtitle.setStyleSheet("font-size: 12px; color: #94a3b8;")
        title_box.addWidget(self.subtitle)
        header.addLayout(title_box)

        header.addStretch()

        # Поисковая строка "на лету"
        self.search_input = QLineEdit(self)
        self.search_input.setPlaceholderText("🔍 Поиск жеста... (например, 'дом', 'друг')")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.setStyleSheet("""
            QLineEdit {
                background-color: #1e293b;
                color: #f8fafc;
                border: 1.5px solid #334155;
                border-radius: 12px;
                padding: 10px 16px;
                font-size: 14px;
                min-width: 320px;
            }
            QLineEdit:focus {
                border-color: #a855f7;
                background-color: #0f172a;
            }
        """)
        self.search_input.textChanged.connect(self._on_search_changed)
        header.addWidget(self.search_input)
        layout.addLayout(header)

        # 2. Горизонтальная панель тематических табов-пилюль (Pill Tabs)
        cat_scroll = QScrollArea(self)
        cat_scroll.setWidgetResizable(True)
        cat_scroll.setFixedHeight(48)
        cat_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        cat_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        cat_scroll.setStyleSheet("background: transparent; border: none;")

        cat_widget = QWidget()
        cat_layout = QHBoxLayout(cat_widget)
        cat_layout.setContentsMargins(0, 2, 0, 2)
        cat_layout.setSpacing(8)

        self.cat_group = QButtonGroup(self)
        self.cat_group.setExclusive(True)

        for cat_id, title_text, icon, _ in CATEGORY_SPECS:
            btn = QPushButton(cat_widget)
            if cat_id == "all":
                btn.setText(f"{icon} {title_text} ({len(self.all_words)})")
            else:
                count = len(self.categories_map.get(cat_id, []))
                btn.setText(f"{icon} {title_text} ({count})" if count else f"{icon} {title_text}")

            btn.setCheckable(True)
            if cat_id == "all":
                btn.setChecked(True)
                btn.setStyleSheet(TAB_ACTIVE_STYLE)
            else:
                btn.setStyleSheet(TAB_INACTIVE_STYLE)

            btn.clicked.connect(lambda checked, cid=cat_id: self._on_category_selected(cid))
            self.cat_group.addButton(btn)
            cat_layout.addWidget(btn)
            self.category_buttons[cat_id] = btn

        cat_layout.addStretch()
        cat_scroll.setWidget(cat_widget)
        layout.addWidget(cat_scroll)

        # 3. Алфавитный рубрикатор (А-Я)
        rubric_scroll = QScrollArea(self)
        rubric_scroll.setWidgetResizable(True)
        rubric_scroll.setFixedHeight(42)
        rubric_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        rubric_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        rubric_scroll.setStyleSheet("background: transparent; border: none;")

        rubric_widget = QWidget()
        rubric_layout = QHBoxLayout(rubric_widget)
        rubric_layout.setContentsMargins(0, 0, 0, 0)
        rubric_layout.setSpacing(5)

        self.rubric_group = QButtonGroup(self)
        self.rubric_group.setExclusive(True)

        btn_all = QPushButton("ВСЕ", rubric_widget)
        btn_all.setCheckable(True)
        btn_all.setChecked(True)
        btn_all.setStyleSheet(RUBRIC_ACTIVE_STYLE)
        btn_all.clicked.connect(lambda: self._on_rubric_selected(None))
        self.rubric_group.addButton(btn_all)
        rubric_layout.addWidget(btn_all)
        self.rubric_buttons["ALL"] = btn_all

        for char in RUSSIAN_ALPHABET:
            has_words = any(
                w.startswith(char) or w.startswith(char.lower()) for w in self.all_words
            )
            btn_char = QPushButton(char, rubric_widget)
            btn_char.setCheckable(True)
            btn_char.setEnabled(has_words)
            btn_char.setStyleSheet(RUBRIC_INACTIVE_STYLE)
            btn_char.clicked.connect(lambda checked, c=char: self._on_rubric_selected(c))
            self.rubric_group.addButton(btn_char)
            rubric_layout.addWidget(btn_char)
            self.rubric_buttons[char] = btn_char

        rubric_layout.addStretch()
        rubric_scroll.setWidget(rubric_widget)
        layout.addWidget(rubric_scroll)

        # 4. Строка статуса фильтрации
        filter_status_layout = QHBoxLayout()
        self.info_counter = QLabel(self)
        self.info_counter.setStyleSheet("font-size: 13px; font-weight: 600; color: #94a3b8;")
        filter_status_layout.addWidget(self.info_counter)
        filter_status_layout.addStretch()
        layout.addLayout(filter_status_layout)

        # 5. Сетка карточек (Grid Cards Scroll Area)
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("background: transparent; border: none;")

        self.grid_container = QWidget()
        self.grid_layout = QGridLayout(self.grid_container)
        self.grid_layout.setContentsMargins(0, 4, 8, 12)
        self.grid_layout.setHorizontalSpacing(14)
        self.grid_layout.setVerticalSpacing(14)

        self.scroll_area.setWidget(self.grid_container)
        layout.addWidget(self.scroll_area, stretch=1)

        # 6. Кнопка "Загрузить ещё" для быстрой пагинации
        self.btn_load_more = QPushButton("Загрузить ещё жесты...", self)
        self.btn_load_more.setProperty("class", "secondary-btn")
        self.btn_load_more.setStyleSheet("""
            QPushButton {
                background-color: #1e293b;
                color: #c084fc;
                font-weight: 700;
                font-size: 14px;
                border: 1.5px solid #7e22ce;
                border-radius: 10px;
                padding: 10px;
                min-height: 40px;
            }
            QPushButton:hover {
                background-color: #2e1065;
                color: #ffffff;
            }
        """)
        self.btn_load_more.clicked.connect(self._on_load_more)
        layout.addWidget(self.btn_load_more)

        self._refresh_word_list()

    def on_enter(self, **kwargs) -> None:
        self._refresh_word_list()

    def _on_category_selected(self, cat_id: str) -> None:
        self.selected_category = cat_id
        for cid, btn in self.category_buttons.items():
            btn.setStyleSheet(TAB_ACTIVE_STYLE if cid == cat_id else TAB_INACTIVE_STYLE)
        self.displayed_count = self.CHUNK_SIZE
        self._refresh_word_list()

    def _on_rubric_selected(self, char: Optional[str]) -> None:
        self.selected_letter = char
        for code, btn in self.rubric_buttons.items():
            is_active = (char is None and code == "ALL") or (code == char)
            btn.setStyleSheet(RUBRIC_ACTIVE_STYLE if is_active else RUBRIC_INACTIVE_STYLE)
        self.displayed_count = self.CHUNK_SIZE
        self._refresh_word_list()

    def _on_search_changed(self, text: str) -> None:
        self.search_query = text.strip().lower()
        self.displayed_count = self.CHUNK_SIZE
        self._refresh_word_list()

    def _on_load_more(self) -> None:
        self.displayed_count += self.CHUNK_SIZE
        self._render_cards()

    def _refresh_word_list(self) -> None:
        """Мгновенная фильтрация и запуск рендеринга."""
        # 1. Фильтр по категории
        if self.selected_category == "all":
            words = list(self.all_words)
        else:
            cat_list = self.categories_map.get(self.selected_category, [])
            words = [w for w in self.all_words if w in cat_list]

        # 2. Фильтр по букве
        if self.selected_letter:
            letter = self.selected_letter.lower()
            words = [w for w in words if w.lower().startswith(letter)]

        # 3. Фильтр по строке поиска
        if self.search_query:
            words = [w for w in words if self.search_query in w.lower()]

        self.filtered_words = words
        self._render_cards()

    def _render_cards(self) -> None:
        """Очистка и заполнение сетки карточек."""
        # Очистка текущих карточек
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        total = len(self.filtered_words)
        showing = min(total, self.displayed_count)

        # Обновление счетчика
        cat_name = dict((c[0], c[1]) for c in CATEGORY_SPECS).get(self.selected_category, "Все")
        filter_desc = f"Категория: «{cat_name}»"
        if self.selected_letter:
            filter_desc += f", Буква: «{self.selected_letter}»"
        if self.search_query:
            filter_desc += f", Поиск: «{self.search_query}»"

        self.info_counter.setText(f"Найдено: {total} жестов ({filter_desc}). Показано: {showing}.")

        if total == 0:
            empty_lbl = QLabel(
                "По заданным критериям жестов не найдено.\nПопробуйте сбросить фильтры или изменить поиск.",
                self.grid_container
            )
            empty_lbl.setStyleSheet("color: #64748b; font-size: 15px; padding: 48px; line-height: 1.6;")
            empty_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.grid_layout.addWidget(empty_lbl, 0, 0, 1, 4)
            self.btn_load_more.hide()
            return

        # Заполнение сетки в 4 колонки
        cols = 4
        words_to_render = self.filtered_words[:showing]

        for idx, word in enumerate(words_to_render):
            category_badge = self.word_badge_map.get(word, "Жест")
            has_skeleton = word.lower() in self.cached_skeletons
            progress = self.db.get_gesture_progress(word)

            card = GestureCardWidget(
                word=word,
                category_name=category_badge,
                has_skeleton=has_skeleton,
                progress=progress,
                on_click=self._open_word_card,
                parent=self.grid_container
            )
            row = idx // cols
            col = idx % cols
            self.grid_layout.addWidget(card, row, col)

        # Управление кнопкой "Загрузить ещё"
        if showing < total:
            remaining = total - showing
            self.btn_load_more.setText(f"Загрузить ещё ({remaining} осталось)...")
            self.btn_load_more.show()
        else:
            self.btn_load_more.hide()

    def _open_word_card(self, word: str) -> None:
        """Переход к детальной карточке жеста для просмотра анимации и тренировки."""
        self.navigator.push_screen("gesture_card_view", word=word)
