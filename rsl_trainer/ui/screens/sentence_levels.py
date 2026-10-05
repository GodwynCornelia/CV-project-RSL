from collections import deque
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCompleter,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget
)

from rsl_trainer.core.audio_feedback import AudioFeedback
from rsl_trainer.core.classifier import TargetGestureVerifier, VerificationResult
from rsl_trainer.cv.camera_worker import CameraWorker
from rsl_trainer.cv.fast_tracker import TrackerMode
from rsl_trainer.cv.ghost_overlay import GhostOverlayRenderer
from rsl_trainer.database.db_manager import DatabaseManager
from rsl_trainer.ui.components.video_canvas import VideoCanvas


class SentenceLevelsScreen(QWidget):
    """
    Экран Б6. Режим «Предложения / Уровни» (по схеме).
    Реализует трехуровневую модель доступа:
    1. Сюжетные уровни (Duolingo) на основе rsl_trainer/data/curriculum.json
    2. Конструктор фраз (Free Phrase Builder) с поиском по 966 классам (QCompleter)
    3. Пошаговый интерактивный тренажер (Guided Composition: [Я] -> [ХОТЕТЬ] -> [ВОДА])
    """

    def __init__(self, navigator, camera_worker: CameraWorker, db: DatabaseManager, parent=None):
        super().__init__(parent)
        self.navigator = navigator
        self.camera_worker = camera_worker
        self.db = db
        self.audio = AudioFeedback()

        self.verifier: Optional[TargetGestureVerifier] = None
        self.ghost_renderer = GhostOverlayRenderer()

        # Данные учебного плана и словаря
        self.curriculum_data = self._load_curriculum()
        self.all_vocabulary = self._load_vocabulary()

        # Состояние конструктора фраз
        self.custom_phrase: List[str] = []

        # Текущая тренируемая фраза
        self.current_title = "УРОВЕНЬ"
        self.current_phrase: List[str] = []
        self.step_idx = 0
        self.frame_buffer = deque(maxlen=16)

        # Состояния тренажера
        self.state = "IDLE"
        self.countdown_val = 3
        self.countdown_timer = QTimer(self)
        self.countdown_timer.setInterval(800)
        self.countdown_timer.timeout.connect(self._on_countdown_tick)

        self._build_ui()

    def _load_curriculum(self) -> Dict:
        """Загрузка curriculum.json."""
        curr_path = Path(__file__).resolve().parent.parent.parent / "data" / "curriculum.json"
        if curr_path.exists():
            try:
                with open(curr_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"tracks": []}

    def _load_vocabulary(self) -> List[str]:
        """Загрузка 966 классов жестов для автодополнения QCompleter."""
        label_file = Path(__file__).resolve().parent.parent.parent.parent / "label2id.json"
        if label_file.exists():
            try:
                with open(label_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    vocab = [w for w in data.keys() if w != "no_event"]
                    vocab.sort()
                    return vocab
            except Exception:
                pass
        return ["я", "хотеть", "вода", "Привет!", "Добро пожаловать!", "дом", "друг", "любить", "время", "день", "хорошо"]

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.stack = QStackedWidget(self)

        # Страница 1: Главный хаб (Сюжетные уровни + Конструктор фраз)
        self.hub_page = QWidget(self)
        self._build_hub_page(self.hub_page)
        self.stack.addWidget(self.hub_page)

        # Страница 2: Интерактивная сборка фразы
        self.trainer_page = QWidget(self)
        self._build_trainer_page(self.trainer_page)
        self.stack.addWidget(self.trainer_page)

        # Страница 3: Финальный экран поздравления
        self.complete_page = QWidget(self)
        self._build_complete_page(self.complete_page)
        self.stack.addWidget(self.complete_page)

        main_layout.addWidget(self.stack)

    # -------------------------------------------------------------
    # СТРАНИЦА 1: ХАБ РАЗДЕЛА «ПРЕДЛОЖЕНИЯ»
    # -------------------------------------------------------------
    def _build_hub_page(self, page: QWidget) -> None:
        layout = QVBoxLayout(page)
        layout.setContentsMargins(32, 20, 32, 24)
        layout.setSpacing(16)

        # 1. Верхняя панель
        header = QHBoxLayout()
        btn_back = QPushButton("← В меню жестов", page)
        btn_back.setProperty("class", "secondary-btn")
        btn_back.clicked.connect(self.navigator.pop_screen)
        header.addWidget(btn_back)

        title = QLabel("КУРС И ПРЕДЛОЖЕНИЯ РЖЯ", page)
        title.setStyleSheet("font-size: 18px; font-weight: 800; color: #10b981;")
        header.addWidget(title)
        header.addStretch()

        self.user_stats_lbl = QLabel(page)
        self.user_stats_lbl.setStyleSheet(
            "background-color: #1e293b; color: #fbbf24; border: 1.5px solid #d97706; "
            "border-radius: 8px; padding: 6px 14px; font-weight: bold; font-size: 13px;"
        )
        header.addWidget(self.user_stats_lbl)
        layout.addLayout(header)

        # 2. Переключатель режимов: Сюжетные уровни / Конструктор фраз
        tab_box = QFrame(page)
        tab_box.setStyleSheet(
            "background-color: #131b2e; border: 1px solid #1e293b; border-radius: 12px; padding: 4px;"
        )
        tab_layout = QHBoxLayout(tab_box)
        tab_layout.setContentsMargins(4, 4, 4, 4)
        tab_layout.setSpacing(8)

        self.mode_group = QButtonGroup(page)
        self.mode_group.setExclusive(True)

        self.btn_mode_story = QPushButton("🗺️ Сюжетные уровни (Duolingo)", tab_box)
        self.btn_mode_story.setCheckable(True)
        self.btn_mode_story.setChecked(True)
        self.btn_mode_story.setProperty("class", "primary-btn")
        self.btn_mode_story.clicked.connect(lambda: self._switch_hub_view(0))
        self.mode_group.addButton(self.btn_mode_story)
        tab_layout.addWidget(self.btn_mode_story)

        self.btn_mode_builder = QPushButton("🛠️ Конструктор фраз (966 жестов)", tab_box)
        self.btn_mode_builder.setCheckable(True)
        self.btn_mode_builder.setProperty("class", "secondary-btn")
        self.btn_mode_builder.clicked.connect(lambda: self._switch_hub_view(1))
        self.mode_group.addButton(self.btn_mode_builder)
        tab_layout.addWidget(self.btn_mode_builder)

        layout.addWidget(tab_box)

        # 3. Контейнер режимов (QStackedWidget)
        self.hub_stack = QStackedWidget(page)

        # Вью 0: Дерево уровней Duolingo
        self.tree_view = QWidget()
        self._build_story_tree_view(self.tree_view)
        self.hub_stack.addWidget(self.tree_view)

        # Вью 1: Конструктор фраз с поиском QCompleter
        self.builder_view = QWidget()
        self._build_phrase_builder_view(self.builder_view)
        self.hub_stack.addWidget(self.builder_view)

        layout.addWidget(self.hub_stack, stretch=1)

    def _switch_hub_view(self, idx: int) -> None:
        self.hub_stack.setCurrentIndex(idx)
        if idx == 0:
            self.btn_mode_story.setProperty("class", "primary-btn")
            self.btn_mode_builder.setProperty("class", "secondary-btn")
        else:
            self.btn_mode_story.setProperty("class", "secondary-btn")
            self.btn_mode_builder.setProperty("class", "primary-btn")
        self.btn_mode_story.style().unpolish(self.btn_mode_story)
        self.btn_mode_story.style().polish(self.btn_mode_story)
        self.btn_mode_builder.style().unpolish(self.btn_mode_builder)
        self.btn_mode_builder.style().polish(self.btn_mode_builder)

    # --- Подстраница 1: Дерево сюжетных уровней ---
    def _build_story_tree_view(self, parent_widget: QWidget) -> None:
        layout = QVBoxLayout(parent_widget)
        layout.setContentsMargins(0, 8, 0, 0)

        scroll = QScrollArea(parent_widget)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background: transparent; border: none;")

        self.levels_container = QWidget()
        self.levels_layout = QVBoxLayout(self.levels_container)
        self.levels_layout.setContentsMargins(20, 10, 20, 20)
        self.levels_layout.setSpacing(16)
        self.levels_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        scroll.setWidget(self.levels_container)
        layout.addWidget(scroll)

    def _refresh_levels_tree(self) -> None:
        stats = self.db.get_user_stats()
        self.user_stats_lbl.setText(f"🔥 {stats['streak_days']} ДНЕЙ  |  ⚡ {stats['total_xp']} XP")

        while self.levels_layout.count():
            item = self.levels_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        tracks = self.curriculum_data.get("tracks", [])
        if not tracks:
            # Фаллбэк из базы данных
            levels = self.db.get_levels()
            for idx, lvl in enumerate(levels):
                card = self._create_db_level_node(lvl, idx)
                self.levels_layout.addWidget(card)
            self.levels_layout.addStretch()
            return

        for t_idx, track in enumerate(tracks):
            track_card = self._create_track_node(track, t_idx)
            self.levels_layout.addWidget(track_card)

            if t_idx < len(tracks) - 1:
                line = QFrame(self.levels_container)
                line.setFixedWidth(4)
                line.setFixedHeight(20)
                line.setStyleSheet("background-color: #10b981; border-radius: 2px;")
                self.levels_layout.addWidget(line, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.levels_layout.addStretch()

    def _create_track_node(self, track: Dict, t_idx: int) -> QWidget:
        node = QFrame(self.levels_container)
        node.setFixedWidth(640)
        node.setStyleSheet(
            "background-color: #1a2332; border: 1.5px solid #2a3b50; border-radius: 16px; padding: 18px 24px;"
        )
        n_layout = QVBoxLayout(node)
        n_layout.setSpacing(12)

        # Заголовок трека
        top_row = QHBoxLayout()
        icon_lbl = QLabel("📚" if t_idx > 0 else "🌟", node)
        icon_lbl.setStyleSheet("font-size: 28px; background: transparent;")
        top_row.addWidget(icon_lbl)

        title_box = QVBoxLayout()
        t_lbl = QLabel(track["title"], node)
        t_lbl.setStyleSheet("font-size: 16px; font-weight: 800; color: #f8fafc; background: transparent;")
        d_lbl = QLabel(track["description"], node)
        d_lbl.setStyleSheet("font-size: 12px; color: #8fa3bf; background: transparent;")
        title_box.addWidget(t_lbl)
        title_box.addWidget(d_lbl)
        top_row.addLayout(title_box, stretch=1)
        n_layout.addLayout(top_row)

        # Список уроков внутри блока
        for lesson in track.get("lessons", []):
            les_row = QFrame(node)
            les_row.setStyleSheet(
                "background-color: #131b2e; border: 1px solid #1e293b; border-radius: 10px; padding: 10px 14px;"
            )
            lr_layout = QHBoxLayout(les_row)
            lr_layout.setContentsMargins(10, 8, 10, 8)
            lr_layout.setSpacing(12)

            lr_text = QVBoxLayout()
            les_title = QLabel(f"🎯 {lesson['title']}: «{lesson['sentence_ru']}»", les_row)
            les_title.setStyleSheet("font-size: 13px; font-weight: bold; color: #38bdf8; background: transparent;")
            chips_str = " ➔ ".join([f"[{g}]" for g in lesson["gloss_sequence"]])
            les_chips = QLabel(f"Жесты: {chips_str}", les_row)
            les_chips.setStyleSheet("font-size: 11px; color: #94a3b8; background: transparent;")
            lr_text.addWidget(les_title)
            lr_text.addWidget(les_chips)
            lr_layout.addLayout(lr_text, stretch=1)

            btn_go = QPushButton("Тренировать ➔", les_row)
            btn_go.setProperty("class", "primary-btn")
            btn_go.clicked.connect(lambda checked, l=lesson: self._start_custom_training(l["title"], l["gloss_sequence"]))
            lr_layout.addWidget(btn_go)

            n_layout.addWidget(les_row)

        return node

    def _create_db_level_node(self, lvl: Dict, idx: int) -> QWidget:
        node = QFrame(self.levels_container)
        node.setFixedWidth(560)
        node.setStyleSheet(
            "background-color: #1a2332; border: 1.5px solid #2a3b50; border-radius: 14px; padding: 16px 20px;"
        )
        l = QHBoxLayout(node)
        t_lbl = QLabel(lvl["title"], node)
        t_lbl.setStyleSheet("font-size: 15px; font-weight: bold; color: #f8fafc;")
        l.addWidget(t_lbl, stretch=1)

        btn = QPushButton("Старт ➔", node)
        btn.setProperty("class", "primary-btn")
        btn.clicked.connect(lambda: self._start_custom_training(lvl["title"], lvl["target_phrase"]))
        l.addWidget(btn)
        return node

    # --- Подстраница 2: Конструктор фраз (Free Phrase Builder) ---
    def _build_phrase_builder_view(self, parent_widget: QWidget) -> None:
        layout = QVBoxLayout(parent_widget)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(16)

        container = QFrame(parent_widget)
        container.setStyleSheet(
            "background-color: #1a2332; border: 1px solid #2a3b50; border-radius: 16px; padding: 24px;"
        )
        c_layout = QVBoxLayout(container)
        c_layout.setSpacing(18)

        # Заголовок
        b_title = QLabel("🛠️ Свободный конструктор фраз (2–4 жеста)", container)
        b_title.setStyleSheet("font-size: 18px; font-weight: 800; color: #f8fafc;")
        b_desc = QLabel(
            "Составьте собственное предложение из базы 966 классов жестов. "
            "Начните вводить слово — умный поиск автоматически предложит варианты.",
            container
        )
        b_desc.setStyleSheet("font-size: 13px; color: #8fa3bf; line-height: 1.4;")
        c_layout.addWidget(b_title)
        c_layout.addWidget(b_desc)

        # Поле ввода с QCompleter
        input_row = QHBoxLayout()
        self.search_input = QLineEdit(container)
        self.search_input.setPlaceholderText("🔍 Введите слово (например: «я», «хотеть», «чай», «вода»)...")
        self.search_input.setStyleSheet("""
            QLineEdit {
                background-color: #131b2e;
                color: #f8fafc;
                border: 1.5px solid #2a3b50;
                border-radius: 10px;
                padding: 10px 16px;
                font-size: 14px;
            }
            QLineEdit:focus {
                border-color: #00b4d8;
            }
        """)

        # Настройка QCompleter
        self.completer = QCompleter(self.all_vocabulary, self)
        self.completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.completer.setFilterMode(Qt.MatchFlag.MatchContains)
        self.completer.setMaxVisibleItems(10)
        self.search_input.setCompleter(self.completer)
        self.search_input.returnPressed.connect(self._add_word_from_input)
        input_row.addWidget(self.search_input, stretch=1)

        btn_add = QPushButton("➕ Добавить жест", container)
        btn_add.setProperty("class", "primary-btn")
        btn_add.clicked.connect(self._add_word_from_input)
        input_row.addWidget(btn_add)
        c_layout.addLayout(input_row)

        # Теги-пилюли (Tags Bar)
        tags_box = QFrame(container)
        tags_box.setStyleSheet(
            "background-color: #131b2e; border: 1.5px dashed #2a3b50; border-radius: 12px; padding: 14px;"
        )
        self.tags_layout = QHBoxLayout(tags_box)
        self.tags_layout.setSpacing(10)
        self.tags_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        c_layout.addWidget(tags_box)

        # Нижняя строка управления
        bottom_row = QHBoxLayout()
        self.phrase_warn = QLabel("💡 Добавьте от 2 до 4 жестов для запуска тренировки", container)
        self.phrase_warn.setStyleSheet("font-size: 12px; color: #94a3b8;")
        bottom_row.addWidget(self.phrase_warn)
        bottom_row.addStretch()

        btn_clear = QPushButton("🗑️ Очистить", container)
        btn_clear.setProperty("class", "secondary-btn")
        btn_clear.clicked.connect(self._clear_custom_phrase)
        bottom_row.addWidget(btn_clear)

        self.btn_start_custom = QPushButton("🚀 Начать тренировку фразы ➔", container)
        self.btn_start_custom.setProperty("class", "primary-btn")
        self.btn_start_custom.setEnabled(False)
        self.btn_start_custom.clicked.connect(self._launch_custom_training)
        bottom_row.addWidget(self.btn_start_custom)

        c_layout.addLayout(bottom_row)

        self._refresh_phrase_tags()

        layout.addWidget(container)
        layout.addStretch()

    def _add_word_from_input(self) -> None:
        text = self.search_input.text().strip()
        if not text:
            return

        # Проверка лимита 4 жеста
        if len(self.custom_phrase) >= 4:
            self.phrase_warn.setText("⚠️ Максимальная длина фразы: 4 жеста.")
            self.phrase_warn.setStyleSheet("font-size: 12px; color: #fbbf24;")
            return

        # Поиск совпадения в словаре (регистронезависимый)
        matched = None
        for v in self.all_vocabulary:
            if v.lower() == text.lower():
                matched = v
                break

        if matched is None:
            # Берем введенное слово, если не найдено строго
            matched = text

        self.custom_phrase.append(matched)
        self.search_input.clear()
        self._refresh_phrase_tags()

    def _remove_word_tag(self, idx: int) -> None:
        if 0 <= idx < len(self.custom_phrase):
            self.custom_phrase.pop(idx)
            self._refresh_phrase_tags()

    def _clear_custom_phrase(self) -> None:
        self.custom_phrase.clear()
        self._refresh_phrase_tags()

    def _refresh_phrase_tags(self) -> None:
        # Очистка
        while self.tags_layout.count():
            item = self.tags_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        if not self.custom_phrase:
            empty_hint = QLabel("Цепочка пуста. Введите слово выше и нажмите «Добавить жест».", self)
            empty_hint.setStyleSheet("color: #64748b; font-size: 13px; font-style: italic;")
            self.tags_layout.addWidget(empty_hint)
            self.btn_start_custom.setEnabled(False)
            self.phrase_warn.setText("💡 Добавьте от 2 до 4 жестов для запуска тренировки")
            self.phrase_warn.setStyleSheet("font-size: 12px; color: #94a3b8;")
            return

        for idx, word in enumerate(self.custom_phrase):
            tag_frame = QFrame()
            tag_frame.setStyleSheet("""
                background-color: #0b1a2d;
                border: 1.5px solid #0077b6;
                border-radius: 8px;
                padding: 4px 10px;
            """)
            tl = QHBoxLayout(tag_frame)
            tl.setContentsMargins(6, 2, 6, 2)
            tl.setSpacing(6)

            tag_lbl = QLabel(word, tag_frame)
            tag_lbl.setStyleSheet("font-size: 13px; font-weight: bold; color: #00b4d8; background: transparent;")
            tl.addWidget(tag_lbl)

            btn_del = QPushButton("✕", tag_frame)
            btn_del.setCursor(Qt.CursorShape.PointingHandCursor)
            btn_del.setStyleSheet("""
                QPushButton {
                    background: transparent;
                    color: #ef4444;
                    font-size: 12px;
                    font-weight: bold;
                    border: none;
                    padding: 0;
                }
                QPushButton:hover {
                    color: #f87171;
                }
            """)
            btn_del.clicked.connect(lambda checked, i=idx: self._remove_word_tag(i))
            tl.addWidget(btn_del)
            self.tags_layout.addWidget(tag_frame)

            if idx < len(self.custom_phrase) - 1:
                arrow_lbl = QLabel("➔", self)
                arrow_lbl.setStyleSheet("color: #475569; font-weight: bold; font-size: 14px;")
                self.tags_layout.addWidget(arrow_lbl)

        # Проверка готовности (2–4 жеста)
        is_ready = 2 <= len(self.custom_phrase) <= 4
        self.btn_start_custom.setEnabled(is_ready)
        if is_ready:
            self.phrase_warn.setText(f"✓ Фраза готова ({len(self.custom_phrase)} жеста). Нажмите «Начать тренировку».")
            self.phrase_warn.setStyleSheet("font-size: 12px; color: #34d399; font-weight: bold;")
        else:
            self.phrase_warn.setText(f"💡 Выбрано: {len(self.custom_phrase)} / 4. Требуется минимум 2 жеста.")
            self.phrase_warn.setStyleSheet("font-size: 12px; color: #94a3b8;")

    def _launch_custom_training(self) -> None:
        title = f"Пользовательская фраза ({len(self.custom_phrase)} жеста)"
        self._start_custom_training(title, self.custom_phrase)

    # -------------------------------------------------------------
    # СТРАНИЦА 2: GUIDED COMPOSITION (ТРЕНИРОВКА ФРАЗЫ)
    # -------------------------------------------------------------
    def _build_trainer_page(self, page: QWidget) -> None:
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 16, 28, 20)
        layout.setSpacing(12)

        # Верхняя панель
        header = QHBoxLayout()
        btn_back = QPushButton("← В карту уровней", page)
        btn_back.setProperty("class", "secondary-btn")
        btn_back.clicked.connect(self._back_to_hub)
        header.addWidget(btn_back)

        self.cur_lvl_title = QLabel("УРОВЕНЬ 1", page)
        self.cur_lvl_title.setStyleSheet("font-size: 18px; font-weight: 800; color: #10b981;")
        header.addWidget(self.cur_lvl_title)
        header.addStretch()

        self.phrase_status = QLabel("Покажите активный жест", page)
        self.phrase_status.setStyleSheet(
            "background-color: #1e293b; color: #fbbf24; border: 1.5px solid #d97706; "
            "border-radius: 8px; padding: 6px 14px; font-weight: bold; font-size: 13px;"
        )
        header.addWidget(self.phrase_status)
        layout.addLayout(header)

        # Интерактивная цепочка чипсов фразы
        self.chips_frame = QFrame(page)
        self.chips_frame.setStyleSheet(
            "background-color: #1e293b; border: 1.5px solid #334155; border-radius: 14px; padding: 10px 18px;"
        )
        self.chips_layout = QHBoxLayout(self.chips_frame)
        self.chips_layout.setSpacing(12)
        layout.addWidget(self.chips_frame)

        # Видео холст
        self.video_canvas = VideoCanvas(page)
        layout.addWidget(self.video_canvas, stretch=1)

        # Нижняя плашка
        b_bar = QFrame(page)
        b_bar.setStyleSheet(
            "background-color: #1e293b; border: 1.5px solid #334155; border-radius: 14px; padding: 12px 20px;"
        )
        bb_layout = QHBoxLayout(b_bar)

        self.phrase_buffer_bar = QProgressBar(b_bar)
        self.phrase_buffer_bar.setRange(0, 16)
        self.phrase_buffer_bar.setValue(0)
        self.phrase_buffer_bar.setTextVisible(False)
        self.phrase_buffer_bar.setFixedWidth(140)
        self.phrase_buffer_bar.setFixedHeight(12)
        bb_layout.addWidget(self.phrase_buffer_bar)

        self.phrase_hint = QLabel("Нажмите «Старт» для отсчета 3-2-1", b_bar)
        self.phrase_hint.setStyleSheet("color: #94a3b8; font-size: 13px;")
        bb_layout.addWidget(self.phrase_hint)
        bb_layout.addStretch()

        self.btn_phrase_action = QPushButton("▶ Начать запись (3-2-1)", b_bar)
        self.btn_phrase_action.setProperty("class", "primary-btn")
        self.btn_phrase_action.clicked.connect(self._toggle_recording)
        bb_layout.addWidget(self.btn_phrase_action)

        layout.addWidget(b_bar)

    # -------------------------------------------------------------
    # СТРАНИЦА 3: ФИНАЛЬНЫЙ ЭКРАН ПОЗДРАВЛЕНИЯ
    # -------------------------------------------------------------
    def _build_complete_page(self, page: QWidget) -> None:
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 32, 40, 32)
        layout.setSpacing(24)
        layout.addStretch()

        card = QFrame(page)
        card.setStyleSheet(
            "background-color: #1e293b; border: 2px solid #10b981; border-radius: 20px; padding: 36px;"
        )
        c_layout = QVBoxLayout(card)
        c_layout.setSpacing(18)

        icon = QLabel("🎉", card)
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet("font-size: 56px; background: transparent;")
        c_layout.addWidget(icon)

        t_lbl = QLabel("ФРАЗА УСПЕШНО ОСВОЕНА!", card)
        t_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        t_lbl.setStyleSheet("font-size: 24px; font-weight: 900; color: #f8fafc;")
        c_layout.addWidget(t_lbl)

        self.complete_stars = QLabel("⭐⭐⭐", card)
        self.complete_stars.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.complete_stars.setStyleSheet("font-size: 32px; background: transparent;")
        c_layout.addWidget(self.complete_stars)

        self.complete_xp = QLabel("+50 XP начислено! Стрик продлен!", card)
        self.complete_xp.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.complete_xp.setStyleSheet("font-size: 16px; font-weight: bold; color: #fbbf24;")
        c_layout.addWidget(self.complete_xp)

        c_layout.addSpacing(16)

        btn_row = QHBoxLayout()
        btn_map = QPushButton("Вернуться в хаб предложений", card)
        btn_map.setProperty("class", "primary-btn")
        btn_map.clicked.connect(self._back_to_hub)
        btn_row.addWidget(btn_map)
        c_layout.addLayout(btn_row)

        layout.addWidget(card)
        layout.addStretch()

    # -------------------------------------------------------------
    # ЛОГИКА И ПЕРЕХОДЫ
    # -------------------------------------------------------------
    def on_enter(self, **kwargs) -> None:
        self.stack.setCurrentIndex(0)
        self._refresh_levels_tree()

    def on_leave(self) -> None:
        self.countdown_timer.stop()
        try:
            self.camera_worker.frame_ready.disconnect(self._on_frame_ready)
        except Exception:
            pass

    def _start_custom_training(self, title: str, gloss_sequence: List[str]) -> None:
        self.current_title = title
        self.current_phrase = list(gloss_sequence)
        self.step_idx = 0
        self.state = "IDLE"
        self.frame_buffer.clear()
        self.phrase_buffer_bar.setValue(0)

        self.cur_lvl_title.setText(title.upper())
        self._render_phrase_chips()

        if self.verifier is None:
            self.phrase_hint.setText("Загрузка нейросети...")
            try:
                self.verifier = TargetGestureVerifier()
            except Exception as e:
                self.phrase_hint.setText(f"Ошибка загрузки: {e}")

        self.stack.setCurrentIndex(1)

        self.camera_worker.set_mode(TrackerMode.DYNAMIC)
        self.camera_worker.frame_ready.connect(self._on_frame_ready)
        if not self.camera_worker.isRunning():
            self.camera_worker.start()

        self._setup_step()

    def _back_to_hub(self) -> None:
        self.on_leave()
        self.stack.setCurrentIndex(0)
        self._refresh_levels_tree()

    def _render_phrase_chips(self) -> None:
        while self.chips_layout.count():
            item = self.chips_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        chip_label_head = QLabel("Фраза:", self.chips_frame)
        chip_label_head.setStyleSheet("font-weight: bold; color: #94a3b8; font-size: 14px;")
        self.chips_layout.addWidget(chip_label_head)

        for i, word in enumerate(self.current_phrase):
            chip = QLabel(self.chips_frame)
            if i < self.step_idx:
                chip.setText(f"✓ {word}")
                chip.setStyleSheet(
                    "background-color: #064e3b; color: #34d399; border: 1.5px solid #059669; "
                    "border-radius: 8px; padding: 6px 14px; font-weight: bold; font-size: 13px;"
                )
            elif i == self.step_idx:
                chip.setText(f"👉 {word}")
                chip.setStyleSheet(
                    "background-color: #451a03; color: #fbbf24; border: 2px solid #f59e0b; "
                    "border-radius: 8px; padding: 6px 14px; font-weight: 800; font-size: 14px;"
                )
            else:
                chip.setText(f"… {word}")
                chip.setStyleSheet(
                    "background-color: #0f172a; color: #64748b; border: 1px dashed #334155; "
                    "border-radius: 8px; padding: 6px 14px; font-size: 13px;"
                )
            self.chips_layout.addWidget(chip)

            if i < len(self.current_phrase) - 1:
                arr = QLabel("➔", self.chips_frame)
                arr.setStyleSheet("color: #475569; font-weight: bold;")
                self.chips_layout.addWidget(arr)

        self.chips_layout.addStretch()

    def _setup_step(self) -> None:
        if self.step_idx >= len(self.current_phrase):
            self._complete_current_level()
            return

        current_target = self.current_phrase[self.step_idx]
        self._render_phrase_chips()
        self.phrase_status.setText(f"Шаг {self.step_idx + 1}/{len(self.current_phrase)}: покажите «{current_target}»")
        self.phrase_status.setStyleSheet(
            "background-color: #1e293b; color: #38bdf8; border: 1.5px solid #0284c7; "
            "border-radius: 8px; padding: 6px 14px; font-weight: bold; font-size: 13px;"
        )
        self.btn_phrase_action.setText("▶ Записать жест (3-2-1)")
        self.btn_phrase_action.setEnabled(True)
        self.phrase_hint.setText(f"Приготовьтесь показать жест «{current_target}»")
        self.state = "IDLE"

    def _toggle_recording(self) -> None:
        if self.state in ("COUNTDOWN", "RECORDING"):
            return

        self.state = "COUNTDOWN"
        self.countdown_val = 3
        self.phrase_status.setText(f"⏱ Приготовьтесь: {self.countdown_val}")
        self.btn_phrase_action.setEnabled(False)
        self.audio.play_countdown()
        self.countdown_timer.start()

    def _on_countdown_tick(self) -> None:
        self.countdown_val -= 1
        if self.countdown_val > 0:
            self.phrase_status.setText(f"⏱ Приготовьтесь: {self.countdown_val}")
            self.audio.play_countdown()
        else:
            self.countdown_timer.stop()
            self._start_recording()

    def _start_recording(self) -> None:
        self.state = "RECORDING"
        self.frame_buffer.clear()
        self.phrase_status.setText("🔴 ИДЕТ ЗАПИСЬ ЖЕСТА...")
        self.phrase_status.setStyleSheet(
            "background-color: #450a0a; color: #f87171; border: 1.5px solid #ef4444; "
            "border-radius: 8px; padding: 6px 14px; font-weight: 800; font-size: 13px;"
        )

    def _on_frame_ready(self, frame_bgr: np.ndarray, features_118: np.ndarray, primary_hand: Optional[np.ndarray]) -> None:
        if self.stack.currentIndex() != 1:
            return

        if self.state == "RECORDING":
            self.frame_buffer.append(features_118.copy())
            self.phrase_buffer_bar.setValue(len(self.frame_buffer))

            if len(self.frame_buffer) >= 16:
                self._evaluate_step()

        frame_with_box, _ = self.ghost_renderer.draw_calibration_box(frame_bgr)
        self.video_canvas.update_frame(frame_with_box)

    def _evaluate_step(self) -> None:
        self.state = "EVALUATING"
        target_word = self.current_phrase[self.step_idx]
        raw_16 = np.array(list(self.frame_buffer))

        if self.verifier is None:
            self.phrase_hint.setText("Модель не загружена")
            self.state = "IDLE"
            self.btn_phrase_action.setEnabled(True)
            return

        res: VerificationResult = self.verifier.verify_target(raw_16, target_word)

        if res.is_passed:
            self.audio.play_success()
            self.step_idx += 1
            if self.step_idx >= len(self.current_phrase):
                self._complete_current_level()
            else:
                self._setup_step()
        else:
            self.audio.play_error()
            self.phrase_status.setText("✗ НЕ ЗАЧТЕНО")
            self.phrase_status.setStyleSheet(
                "background-color: #450a0a; color: #f87171; border: 1.5px solid #ef4444; "
                "border-radius: 8px; padding: 6px 14px; font-weight: 800; font-size: 13px;"
            )
            self.phrase_hint.setText(f"{res.message}")
            self.btn_phrase_action.setText("🔄 Попробовать снова")
            self.btn_phrase_action.setEnabled(True)
            self.state = "IDLE"

    def _complete_current_level(self) -> None:
        self.on_leave()
        self.db.add_xp(50)
        self.db.update_streak()
        self.audio.play_level_complete()
        self.stack.setCurrentIndex(2)
