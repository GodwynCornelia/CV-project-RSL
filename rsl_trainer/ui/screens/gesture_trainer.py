from collections import deque
from pathlib import Path
from typing import Optional
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget
)

from rsl_trainer.config import SKELETONS_DIR
from rsl_trainer.core.audio_feedback import AudioFeedback
from rsl_trainer.core.classifier import TargetGestureVerifier, VerificationResult
from rsl_trainer.cv.camera_worker import CameraWorker
from rsl_trainer.cv.fast_tracker import TrackerMode
from rsl_trainer.cv.ghost_overlay import GhostOverlayRenderer
from rsl_trainer.database.db_manager import DatabaseManager
from rsl_trainer.ui.components.video_canvas import VideoCanvas


class GestureTrainerScreen(QWidget):
    """
    Экран Б5. Тренажер динамического жеста (по схеме).
    - Камера в режиме DYNAMIC (YOLO-pose 320x240 + MediaPipe Hands)
    - Скользящий циклический буфер 16 кадров (16 x 118)
    - Режим записи по таймеру «3-2-1-Старт» или автоматический скользящий инференс
    - Проверка через Target-Constrained Verifier (Top-3 / P >= 0.30)
    - Начисление XP и фиксация в БД
    """

    def __init__(self, navigator, camera_worker: CameraWorker, db: DatabaseManager, parent=None):
        super().__init__(parent)
        self.navigator = navigator
        self.camera_worker = camera_worker
        self.db = db
        self.audio = AudioFeedback()

        self.verifier: Optional[TargetGestureVerifier] = None
        self.ghost_renderer = GhostOverlayRenderer()

        self.target_word = "Привет!"
        self.frame_buffer = deque(maxlen=16)
        self.template_trajectory: Optional[np.ndarray] = None

        # Состояния тренажера: IDLE, COUNTDOWN, RECORDING, EVALUATING, SUCCESS
        self.state = "IDLE"
        self.countdown_val = 3

        self.countdown_timer = QTimer(self)
        self.countdown_timer.setInterval(800)
        self.countdown_timer.timeout.connect(self._on_countdown_tick)

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 16, 28, 20)
        layout.setSpacing(12)

        # Верхняя панель
        header = QHBoxLayout()
        btn_back = QPushButton("← В словарь", self)
        btn_back.setProperty("class", "secondary-btn")
        btn_back.clicked.connect(self._go_back)
        header.addWidget(btn_back)

        self.title_lbl = QLabel("ТРЕНИРОВКА: «ПРИВЕТ!»", self)
        self.title_lbl.setStyleSheet("font-size: 18px; font-weight: 800; color: #a855f7;")
        header.addWidget(self.title_lbl)

        header.addStretch()

        self.status_pill = QLabel("Готов к записи", self)
        self.status_pill.setStyleSheet(
            "background-color: #1e293b; color: #38bdf8; border: 1.5px solid #0284c7; "
            "border-radius: 8px; padding: 6px 16px; font-size: 14px; font-weight: bold;"
        )
        header.addWidget(self.status_pill)
        layout.addLayout(header)

        # Холст камеры
        self.video_canvas = VideoCanvas(self)
        layout.addWidget(self.video_canvas, stretch=1)

        # Нижняя плашка управления
        bottom_bar = QFrame(self)
        bottom_bar.setStyleSheet(
            "background-color: #1e293b; border: 1.5px solid #334155; border-radius: 14px; padding: 12px 20px;"
        )
        b_layout = QHBoxLayout(bottom_bar)
        b_layout.setSpacing(16)

        # Индикатор накопления 16 кадров
        self.buffer_bar = QProgressBar(bottom_bar)
        self.buffer_bar.setRange(0, 16)
        self.buffer_bar.setValue(0)
        self.buffer_bar.setTextVisible(False)
        self.buffer_bar.setFixedWidth(140)
        self.buffer_bar.setFixedHeight(12)
        b_layout.addWidget(self.buffer_bar)

        self.hint_lbl = QLabel("Нажмите «Старт» или Пробел для запуска отсчета 3-2-1", bottom_bar)
        self.hint_lbl.setStyleSheet("color: #94a3b8; font-size: 13px;")
        b_layout.addWidget(self.hint_lbl)

        b_layout.addStretch()

        self.btn_action = QPushButton("▶ Начать запись (3-2-1)", bottom_bar)
        self.btn_action.setProperty("class", "primary-btn")
        self.btn_action.clicked.connect(self._toggle_recording)
        b_layout.addWidget(self.btn_action)

        layout.addWidget(bottom_bar)

    def on_enter(self, target_word: str = "Привет!", **kwargs) -> None:
        self.target_word = target_word
        self.title_lbl.setText(f"ТРЕНИРОВКА: «{target_word.upper()}»")
        self.state = "IDLE"
        self.frame_buffer.clear()
        self.buffer_bar.setValue(0)
        self.status_pill.setText("Готов к записи")
        self.status_pill.setStyleSheet(
            "background-color: #1e293b; color: #38bdf8; border: 1.5px solid #0284c7; "
            "border-radius: 8px; padding: 6px 16px; font-size: 14px; font-weight: bold;"
        )
        self.btn_action.setText("▶ Начать запись (3-2-1)")
        self.btn_action.setEnabled(True)

        # Ленивая загрузка верификатора при первом открытии
        if self.verifier is None:
            self.hint_lbl.setText("Загрузка нейросети...")
            try:
                self.verifier = TargetGestureVerifier()
            except Exception as e:
                self.hint_lbl.setText(f"Ошибка загрузки модели: {e}")

        self.hint_lbl.setText("Нажмите «Старт» для отсчета 3-2-1 и покажите жест")

        # Поиск эталонного скелета
        npy_path = SKELETONS_DIR / f"{target_word}.npy"
        if not npy_path.exists():
            npy_path = SKELETONS_DIR / f"{target_word.lower()}.npy"
        if npy_path.exists():
            try:
                self.template_trajectory = np.load(npy_path)
            except Exception:
                self.template_trajectory = None
        else:
            self.template_trajectory = None

        # Переключаем трекер в режим DYNAMIC
        self.camera_worker.set_mode(TrackerMode.DYNAMIC)
        self.camera_worker.frame_ready.connect(self._on_frame_ready)

        if not self.camera_worker.isRunning():
            self.camera_worker.start()

    def on_leave(self) -> None:
        self.countdown_timer.stop()
        try:
            self.camera_worker.frame_ready.disconnect(self._on_frame_ready)
        except Exception:
            pass

    def _go_back(self) -> None:
        self.on_leave()
        self.navigator.pop_screen()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space:
            self._toggle_recording()
        else:
            super().keyPressEvent(event)

    def _toggle_recording(self) -> None:
        if self.state in ("COUNTDOWN", "RECORDING"):
            return

        self.state = "COUNTDOWN"
        self.countdown_val = 3
        self.status_pill.setText(f"⏱ Приготовьтесь: {self.countdown_val}")
        self.status_pill.setStyleSheet(
            "background-color: #451a03; color: #fbbf24; border: 1.5px solid #d97706; "
            "border-radius: 8px; padding: 6px 16px; font-size: 15px; font-weight: 800;"
        )
        self.btn_action.setEnabled(False)
        self.audio.play_countdown()
        self.countdown_timer.start()

    def _on_countdown_tick(self) -> None:
        self.countdown_val -= 1
        if self.countdown_val > 0:
            self.status_pill.setText(f"⏱ Приготовьтесь: {self.countdown_val}")
            self.audio.play_countdown()
        else:
            self.countdown_timer.stop()
            self._start_recording()

    def _start_recording(self) -> None:
        self.state = "RECORDING"
        self.frame_buffer.clear()
        self.status_pill.setText("🔴 ИДЕТ ЗАПИСЬ ЖЕСТА...")
        self.status_pill.setStyleSheet(
            "background-color: #450a0a; color: #f87171; border: 1.5px solid #ef4444; "
            "border-radius: 8px; padding: 6px 16px; font-size: 15px; font-weight: 800;"
        )
        self.hint_lbl.setText("Выполняйте движение! Фиксируем 16 кадров...")

    def _on_frame_ready(self, frame_bgr: np.ndarray, features_118: np.ndarray, primary_hand: Optional[np.ndarray]) -> None:
        # 1. Если идет запись, добавляем вектор признаков
        if self.state == "RECORDING":
            self.frame_buffer.append(features_118.copy())
            self.buffer_bar.setValue(len(self.frame_buffer))

            if len(self.frame_buffer) >= 16:
                self._evaluate_gesture()

        # 2. Рендеринг кадра с калибровочной рамкой
        frame_with_box, _ = self.ghost_renderer.draw_calibration_box(frame_bgr)
        self.video_canvas.update_frame(frame_with_box)

    def _evaluate_gesture(self) -> None:
        self.state = "EVALUATING"
        raw_16 = np.array(list(self.frame_buffer))  # (16, 118)

        if self.verifier is None:
            self.hint_lbl.setText("Модель не инициализирована")
            self._reset_state()
            return

        # Инференс через Target-Constrained Verifier
        res: VerificationResult = self.verifier.verify_target(raw_16, self.target_word)

        if res.is_passed:
            self.state = "SUCCESS"
            self.audio.play_success()

            # Обновление базы данных
            self.db.update_gesture_progress(self.target_word, res.target_confidence)
            self.db.add_xp(30)
            self.db.update_streak()

            self.status_pill.setText(f"✓ ПРИНЯТО! (Ранг #{res.target_rank}, {res.target_confidence*100:.0f}%)")
            self.status_pill.setStyleSheet(
                "background-color: #064e3b; color: #34d399; border: 1.5px solid #059669; "
                "border-radius: 8px; padding: 6px 16px; font-size: 15px; font-weight: 800;"
            )
            self.hint_lbl.setText("🎉 Отлично! Жест успешно освоен (+30 XP)")
            self.btn_action.setText("🔄 Повторить жест")
            self.btn_action.setEnabled(True)
        else:
            self.audio.play_error()
            self.status_pill.setText("✗ НЕ ЗАЧТЕНО")
            self.status_pill.setStyleSheet(
                "background-color: #450a0a; color: #f87171; border: 1.5px solid #ef4444; "
                "border-radius: 8px; padding: 6px 16px; font-size: 15px; font-weight: 800;"
            )
            self.hint_lbl.setText(f"{res.message}")
            self.btn_action.setText("🔄 Попробовать снова")
            self.btn_action.setEnabled(True)
            self.state = "IDLE"

    def _reset_state(self) -> None:
        self.state = "IDLE"
        self.btn_action.setEnabled(True)
        self.btn_action.setText("▶ Начать запись (3-2-1)")
