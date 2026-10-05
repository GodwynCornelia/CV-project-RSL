from collections import deque
import json
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
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
    Чистое линейное дерево уровней Duolingo и пошаговая сборка фраз (Guided Composition):
    - Дерево уровней: Уровень 1 ➔ Уровень 2 ➔ Уровень 3...
    - Внутри уровня — последовательная валидация жестов заданной фразы
    - Начисление XP (+50 XP), обновление Streak и открытие следующих уровней
    """

    def __init__(self, navigator, camera_worker: CameraWorker, db: DatabaseManager, parent=None):
        super().__init__(parent)
        self.navigator = navigator
        self.camera_worker = camera_worker
        self.db = db
        self.audio = AudioFeedback()

        self.verifier: Optional[TargetGestureVerifier] = None
        self.ghost_renderer = GhostOverlayRenderer()

        # Текущий уровень и фраза
        self.current_level_data: Optional[Dict] = None
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

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.stack = QStackedWidget(self)

        # 1. Дерево уровней Duolingo
        self.tree_page = QWidget(self)
        self._build_tree_page(self.tree_page)
        self.stack.addWidget(self.tree_page)

        # 2. Интерактивная сборка фразы (Guided Composition)
        self.trainer_page = QWidget(self)
        self._build_trainer_page(self.trainer_page)
        self.stack.addWidget(self.trainer_page)

        # 3. Финальный экран поздравления
        self.complete_page = QWidget(self)
        self._build_complete_page(self.complete_page)
        self.stack.addWidget(self.complete_page)

        main_layout.addWidget(self.stack)

    # -------------------------------------------------------------
    # СТРАНИЦА 1: ДЕРЕВО УРОВНЕЙ DUOLINGO
    # -------------------------------------------------------------
    def _build_tree_page(self, page: QWidget) -> None:
        layout = QVBoxLayout(page)
        layout.setContentsMargins(32, 20, 32, 24)
        layout.setSpacing(16)

        # Верхняя панель
        header = QHBoxLayout()
        btn_back = QPushButton("← В меню жестов", page)
        btn_back.setProperty("class", "secondary-btn")
        btn_back.clicked.connect(self.navigator.pop_screen)
        header.addWidget(btn_back)

        title = QLabel("КУРС РЖЯ: ДЕРЕВО УРОВНЕЙ", page)
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

        # Прокручиваемый список уровней
        scroll = QScrollArea(page)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setStyleSheet("background: transparent; border: none;")

        self.levels_container = QWidget()
        self.levels_layout = QVBoxLayout(self.levels_container)
        self.levels_layout.setContentsMargins(20, 10, 20, 20)
        self.levels_layout.setSpacing(16)
        self.levels_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        scroll.setWidget(self.levels_container)
        layout.addWidget(scroll, stretch=1)

    def _refresh_levels_tree(self) -> None:
        stats = self.db.get_user_stats()
        self.user_stats_lbl.setText(f"🔥 {stats['streak_days']} ДНЕЙ  |  ⚡ {stats['total_xp']} XP")

        # Очистка
        while self.levels_layout.count():
            item = self.levels_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        levels = self.db.get_levels()
        for idx, lvl in enumerate(levels):
            card = self._create_level_node(lvl, idx)
            self.levels_layout.addWidget(card)

            if idx < len(levels) - 1:
                line = QFrame(self.levels_container)
                line.setFixedWidth(4)
                line.setFixedHeight(24)
                line_color = "#10b981" if lvl["stars"] > 0 else "#334155"
                line.setStyleSheet(f"background-color: {line_color}; border-radius: 2px;")
                self.levels_layout.addWidget(line, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.levels_layout.addStretch()

    def _create_level_node(self, lvl: Dict, idx: int) -> QWidget:
        node = QFrame(self.levels_container)
        node.setFixedWidth(560)

        is_unlocked = bool(lvl["is_unlocked"])
        stars = lvl["stars"]

        bg_color = "#1a2332" if is_unlocked else "#0f172a"
        border_color = "#10b981" if stars == 3 else ("#0284c7" if is_unlocked else "#2a3b50")

        node.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_color};
                border: 2px solid {border_color};
                border-radius: 16px;
                padding: 16px 22px;
            }}
        """)

        n_layout = QHBoxLayout(node)
        n_layout.setSpacing(16)

        # Иконка узла
        icon_lbl = QLabel(node)
        icon_lbl.setStyleSheet("font-size: 34px; background: transparent; border: none;")
        if not is_unlocked:
            icon_lbl.setText("🔒")
        elif stars == 3:
            icon_lbl.setText("👑")
        else:
            icon_lbl.setText("⭐")
        n_layout.addWidget(icon_lbl)

        # Текст уровня
        info_layout = QVBoxLayout()
        info_layout.setSpacing(4)
        t_lbl = QLabel(lvl["title"], node)
        t_color = "#f8fafc" if is_unlocked else "#64748b"
        t_lbl.setStyleSheet(f"font-size: 16px; font-weight: 800; color: {t_color}; background: transparent; border: none;")
        info_layout.addWidget(t_lbl)

        d_lbl = QLabel(lvl["description"], node)
        d_lbl.setStyleSheet("font-size: 13px; color: #8fa3bf; background: transparent; border: none;")
        info_layout.addWidget(d_lbl)

        # Цепочка слов фразы
        phrase_chips = "  ➔  ".join([f"[{w}]" for w in lvl["target_phrase"]])
        p_lbl = QLabel(f"Фраза: {phrase_chips}", node)
        p_lbl.setStyleSheet("font-size: 12px; color: #38bdf8; font-weight: bold; background: transparent; border: none;")
        info_layout.addWidget(p_lbl)

        n_layout.addLayout(info_layout, stretch=1)

        # Правый блок: Звезды и кнопка старта
        right_box = QVBoxLayout()
        right_box.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_box.setSpacing(6)

        stars_str = "⭐" * stars + "☆" * (3 - stars)
        stars_lbl = QLabel(stars_str if is_unlocked else "", node)
        stars_lbl.setStyleSheet("font-size: 13px; background: transparent; border: none;")
        right_box.addWidget(stars_lbl)

        btn_start = QPushButton("Старт ➔" if is_unlocked else "Закрыт", node)
        btn_start.setProperty("class", "primary-btn" if is_unlocked else "secondary-btn")
        btn_start.setEnabled(is_unlocked)
        btn_start.clicked.connect(lambda checked, l=lvl: self._start_level(l))
        right_box.addWidget(btn_start)

        n_layout.addLayout(right_box)
        return node

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
        btn_back.clicked.connect(self._back_to_tree)
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
            "background-color: #1a2332; border: 1px solid #2a3b50; border-radius: 14px; padding: 10px 18px;"
        )
        self.chips_layout = QHBoxLayout(self.chips_frame)
        self.chips_layout.setSpacing(12)
        layout.addWidget(self.chips_frame)

        # Видео холст
        self.video_canvas = VideoCanvas(page)
        layout.addWidget(self.video_canvas, stretch=1)

        # Нижняя контрольная плашка
        b_bar = QFrame(page)
        b_bar.setStyleSheet(
            "background-color: #1a2332; border: 1px solid #2a3b50; border-radius: 14px; padding: 12px 20px;"
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
        self.phrase_hint.setStyleSheet("color: #8fa3bf; font-size: 13px;")
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
            "background-color: #1a2332; border: 2px solid #10b981; border-radius: 20px; padding: 36px;"
        )
        c_layout = QVBoxLayout(card)
        c_layout.setSpacing(18)

        icon = QLabel("🎉", card)
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet("font-size: 56px; background: transparent; border: none;")
        c_layout.addWidget(icon)

        t_lbl = QLabel("УРОВЕНЬ УСПЕШНО ПРОЙДЕН!", card)
        t_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        t_lbl.setStyleSheet("font-size: 24px; font-weight: 900; color: #f8fafc; border: none;")
        c_layout.addWidget(t_lbl)

        self.complete_stars = QLabel("⭐⭐⭐", card)
        self.complete_stars.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.complete_stars.setStyleSheet("font-size: 32px; background: transparent; border: none;")
        c_layout.addWidget(self.complete_stars)

        self.complete_xp = QLabel("+50 XP начислено! Стрик продлен!", card)
        self.complete_xp.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.complete_xp.setStyleSheet("font-size: 16px; font-weight: bold; color: #fbbf24; border: none;")
        c_layout.addWidget(self.complete_xp)

        c_layout.addSpacing(16)

        btn_row = QHBoxLayout()
        btn_map = QPushButton("Вернуться к карте уровней", card)
        btn_map.setProperty("class", "primary-btn")
        btn_map.clicked.connect(self._back_to_tree)
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

    def _start_level(self, level_dict: Dict) -> None:
        self.current_level_data = level_dict
        self.current_phrase = list(level_dict["target_phrase"])
        self.step_idx = 0
        self.state = "IDLE"
        self.frame_buffer.clear()
        self.phrase_buffer_bar.setValue(0)

        self.cur_lvl_title.setText(level_dict["title"].upper())
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

    def _back_to_tree(self) -> None:
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
        chip_label_head.setStyleSheet("font-weight: bold; color: #8fa3bf; font-size: 14px; border: none;")
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
                arr.setStyleSheet("color: #475569; font-weight: bold; border: none;")
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
        level_id = self.current_level_data.get("level_id", 1)
        self.db.complete_level(level_id, stars=3)
        self.db.add_xp(50)
        self.db.update_streak()
        self.audio.play_level_complete()
        self.stack.setCurrentIndex(2)
