from typing import Optional
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget
)

from rsl_trainer.core.audio_feedback import AudioFeedback
from rsl_trainer.core.dactyl_evaluator import DactylEvaluator
from rsl_trainer.cv.camera_worker import CameraWorker
from rsl_trainer.cv.fast_tracker import TrackerMode
from rsl_trainer.cv.ghost_overlay import GhostOverlayRenderer
from rsl_trainer.database.db_manager import DatabaseManager, RUSSIAN_ALPHABET
from rsl_trainer.ui.components.video_canvas import VideoCanvas


class DactylTrainerScreen(QWidget):
    """
    Экран А4. Тренажер буквы (Интерактивная камера по схеме).
    - Окно камеры с Ghost Overlay (полупрозрачный каркас кисти)
    - Габаритная рамка калибровки
    - Шкала соответствия и 1.5-секундный таймер удержания
    - Автоматическая фиксация сдачи в базе данных (+25 XP)
    """

    def __init__(self, navigator, camera_worker: CameraWorker, db: DatabaseManager, parent=None):
        super().__init__(parent)
        self.navigator = navigator
        self.camera_worker = camera_worker
        self.db = db
        self.audio = AudioFeedback(config=camera_worker.config)
        self._active = False
        camera_worker.camera_error.connect(self._on_camera_error)

        self.current_letter = "А"
        self.evaluator = DactylEvaluator(skeletons_dir=camera_worker.config.skeletons_dir,
                                         threshold=camera_worker.config.dactyl_threshold,
                                         hold_duration=camera_worker.config.dactyl_hold_seconds)
        self.ghost_renderer = GhostOverlayRenderer()
        self.template_hand: Optional[np.ndarray] = None
        self._is_completed = False

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 16, 28, 20)
        layout.setSpacing(12)

        # Верхняя панель
        header = QHBoxLayout()
        btn_back = QPushButton("← В карточку", self)
        btn_back.setProperty("class", "secondary-btn")
        btn_back.clicked.connect(self._go_back)
        header.addWidget(btn_back)

        self.title_lbl = QLabel("ТРЕНИРОВКА: БУКВА «А»", self)
        self.title_lbl.setStyleSheet("font-size: 18px; font-weight: 800; color: #38bdf8;")
        header.addWidget(self.title_lbl)

        header.addStretch()

        self.status_pill = QLabel("Удерживайте позу 1.5 сек", self)
        self.status_pill.setStyleSheet(
            "background-color: #1e293b; color: #fbbf24; border: 1.5px solid #d97706; "
            "border-radius: 8px; padding: 6px 14px; font-weight: bold;"
        )
        header.addWidget(self.status_pill)
        layout.addLayout(header)

        # Экранный холст видео
        self.video_canvas = VideoCanvas(self)
        layout.addWidget(self.video_canvas, stretch=1)

        # Нижняя контрольная панель
        bottom_bar = QHBoxLayout()

        self.info_hint = QLabel("💡 Расположите руку в рамке так, чтобы повторить полупрозрачный каркас.", self)
        self.info_hint.setStyleSheet("color: #94a3b8; font-size: 13px;")
        bottom_bar.addWidget(self.info_hint)

        bottom_bar.addStretch()

        self.btn_next = QPushButton("Следующая буква ➔", self)
        self.btn_next.setProperty("class", "primary-btn")
        self.btn_next.setEnabled(False)
        self.btn_next.clicked.connect(self._go_next_letter)
        bottom_bar.addWidget(self.btn_next)

        layout.addLayout(bottom_bar)

    def on_enter(self, letter: str = "А", **kwargs) -> None:
        """Запуск камеры и установка целевой буквы."""
        self.current_letter = letter
        self._active = True
        self._is_completed = False
        self.btn_next.setEnabled(False)
        self.title_lbl.setText(f"ТРЕНИРОВКА: БУКВА «{letter}»")
        self.status_pill.setText("Удерживайте позу 1.5 сек")
        self.status_pill.setStyleSheet(
            "background-color: #1e293b; color: #fbbf24; border: 1.5px solid #d97706; "
            "border-radius: 8px; padding: 6px 14px; font-weight: bold;"
        )

        self.evaluator.set_target_letter(letter)
        self.template_hand = self.evaluator._get_template_hand(letter)

        # Переключаем трекер в быстрый режим DACTYL (только руки, 55+ FPS)
        self.camera_worker.set_mode(TrackerMode.DACTYL)
        self.camera_worker.frame_ready.connect(self._on_frame_ready)

        if not self.camera_worker.isRunning():
            self.camera_worker.start()

    def on_leave(self) -> None:
        """Отключение сигналов при уходе с экрана."""
        self._active = False
        try:
            self.camera_worker.frame_ready.disconnect(self._on_frame_ready)
        except Exception:
            pass
        self.evaluator.evaluate_hand(None)
        self.camera_worker.stop()

    def _go_back(self) -> None:
        self.navigator.pop_screen()

    def _on_frame_ready(self, frame_bgr: np.ndarray, features_118: np.ndarray, primary_hand: Optional[np.ndarray]) -> None:
        if not self._active:
            return
        if self._is_completed:
            # Отрисовка финального кадра
            self.video_canvas.update_frame(frame_bgr)
            return

        # 1. Отрисовка калибровочной рамки
        frame_with_box, calib_box = self.ghost_renderer.draw_calibration_box(frame_bgr)

        # 2. Отрисовка Ghost Overlay эталонной кисти
        if self.template_hand is not None:
            frame_with_box = self.ghost_renderer.render_ghost_hand(
                frame_with_box,
                template_hand_21x2=self.template_hand,
                live_hand_21x2=primary_hand,
                calibration_box=calib_box
            )

        # 3. Оценка совпадения и удержания
        score, hold_prog, is_confirmed = self.evaluator.evaluate_hand(primary_hand)

        self.video_canvas.set_hold_metrics(score, hold_prog, show_bar=True)
        self.video_canvas.update_frame(frame_with_box)

        # 4. Обработка успешного выполнения
        if is_confirmed and not self._is_completed:
            self._on_success(score)

    def _on_camera_error(self, message):
        if self._active:
            self.evaluator.evaluate_hand(None)
            self.video_canvas.set_placeholder_text("Переподключите камеру, чтобы продолжить тренировку")

    def _on_success(self, score: float) -> None:
        self._is_completed = True
        self.audio.play_success()

        # Запись в базу данных
        self.db.update_dactyl_progress(self.current_letter, score)
        new_xp = self.db.add_xp(25)
        self.db.update_streak()

        self.status_pill.setText(f"✓ ЗАЧТЕНО! +25 XP ({score*100:.0f}%)")
        self.status_pill.setStyleSheet(
            "background-color: #064e3b; color: #34d399; border: 1.5px solid #059669; "
            "border-radius: 8px; padding: 6px 14px; font-weight: bold; font-size: 14px;"
        )
        self.btn_next.setEnabled(True)

    def _go_next_letter(self) -> None:
        try:
            curr_idx = RUSSIAN_ALPHABET.index(self.current_letter)
            if curr_idx + 1 < len(RUSSIAN_ALPHABET):
                next_l = RUSSIAN_ALPHABET[curr_idx + 1]
                self.on_leave()
                self.on_enter(letter=next_l)
            else:
                self._go_back()
        except ValueError:
            self._go_back()
