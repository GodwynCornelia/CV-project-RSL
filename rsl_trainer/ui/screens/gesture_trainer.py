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

from rsl_trainer.core.recording import GestureRecording
from rsl_trainer.core.trajectories import find_trajectory
from rsl_trainer.core.audio_feedback import AudioFeedback
from rsl_trainer.core.classifier import VerificationResult
from rsl_trainer.cv.camera_worker import CameraWorker
from rsl_trainer.cv.fast_tracker import TrackerMode
from rsl_trainer.cv.ghost_overlay import GhostOverlayRenderer
from rsl_trainer.database.db_manager import DatabaseManager
from rsl_trainer.ui.components.video_canvas import VideoCanvas
from rsl_trainer.ui.components.gesture_preview import GestureSilhouettePreview


class GestureTrainerScreen(QWidget):
    """
    Экран Б5. Тренажер динамического жеста (по схеме).
    - Камера в режиме DYNAMIC (YOLO-pose 320x240 + MediaPipe Hands)
    - Запись полного движения и выборка 16 фаз по времени
    - Отсчет «3-2-1-Старт» и фоновое распознавание
    - Проверка через Target-Constrained Verifier (Top-3 / P >= 0.30)
    - Начисление XP и фиксация в БД
    """

    def __init__(self, navigator, camera_worker: CameraWorker, db: DatabaseManager, parent=None):
        super().__init__(parent)
        self.navigator = navigator
        self.camera_worker = camera_worker
        self.db = db
        self.audio = AudioFeedback(config=camera_worker.config)

        self.inference = camera_worker.inference_service
        self.inference.loaded.connect(self._on_model_loaded)
        self.inference.verified.connect(self._on_verified)
        self.inference.failed.connect(self._on_inference_error)
        self._active = False
        self._request_id = None
        self._model_ready = False
        self._camera_failed = False
        camera_worker.camera_error.connect(self._on_camera_error)
        self.ghost_renderer = GhostOverlayRenderer()

        self.target_word = "Привет!"
        self.recording = GestureRecording(camera_worker.config.dynamic_recording_seconds)
        self.template_trajectory: Optional[np.ndarray] = None

        # Состояния тренажера: IDLE, COUNTDOWN, RECORDING, EVALUATING, SUCCESS
        self.state = "IDLE"
        self.countdown_val = 3

        self.countdown_timer = QTimer(self)
        self.countdown_timer.setInterval(1000)
        self.countdown_timer.timeout.connect(self._on_countdown_tick)
        self.recording_timer = QTimer(self)
        self.recording_timer.setSingleShot(True)
        self.recording_timer.timeout.connect(self._evaluate_gesture)

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
        self.title_lbl.setWordWrap(True)
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
        self.video_canvas.setMinimumHeight(200)
        layout.addWidget(self.video_canvas, stretch=1)
        self.preview_widget = GestureSilhouettePreview(self, config=self.camera_worker.config)
        self.preview_widget.setFixedHeight(240)
        layout.addWidget(self.preview_widget)

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
        self.hint_lbl.setWordWrap(True)
        self.hint_lbl.setStyleSheet("color: #94a3b8; font-size: 13px;")
        b_layout.addWidget(self.hint_lbl)

        b_layout.addStretch()

        self.btn_action = QPushButton("▶ Начать запись (3-2-1)", bottom_bar)
        self.btn_action.setProperty("class", "primary-btn")
        self.btn_action.clicked.connect(self._toggle_recording)
        b_layout.addWidget(self.btn_action)

        layout.addWidget(bottom_bar)

    def on_enter(self, target_word: str = "Привет!", **kwargs) -> None:
        self._active = True
        self._camera_failed = False
        self.recording.duration = self.camera_worker.config.dynamic_recording_seconds
        self.target_word = target_word
        self.title_lbl.setText(f"ТРЕНИРОВКА: «{target_word.upper()}»")
        self.state = "IDLE"
        self.recording.start()
        self.buffer_bar.setValue(0)
        self.status_pill.setText("Готов к записи")
        self.status_pill.setStyleSheet(
            "background-color: #1e293b; color: #38bdf8; border: 1.5px solid #0284c7; "
            "border-radius: 8px; padding: 6px 16px; font-size: 14px; font-weight: bold;"
        )
        self.btn_action.setText("▶ Начать запись (3-2-1)")
        self.btn_action.setEnabled(False)

        self._model_ready = False
        self.hint_lbl.setText("Загрузка нейросети...")
        self._request_id = self.inference.load()
        self.template_trajectory = find_trajectory(target_word, self.camera_worker.config)
        self.preview_widget.set_trajectory(self.template_trajectory)

        # Переключаем трекер в режим DYNAMIC
        self.camera_worker.set_mode(TrackerMode.DYNAMIC)
        self.camera_worker.frame_ready.connect(self._on_frame_ready)

        if not self.camera_worker.isRunning():
            self.camera_worker.start()

    def on_leave(self) -> None:
        self._active = False
        self._request_id = None
        self.countdown_timer.stop()
        self.recording_timer.stop()
        self.preview_widget.finish_recording()
        self.preview_widget.stop_animation()
        try:
            self.camera_worker.frame_ready.disconnect(self._on_frame_ready)
        except Exception:
            pass
        self.camera_worker.stop()

    def _go_back(self) -> None:
        self.navigator.pop_screen()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Space:
            self._toggle_recording()
        else:
            super().keyPressEvent(event)

    def _toggle_recording(self) -> None:
        if not self._active or not self._model_ready or self._camera_failed or self.state in ("COUNTDOWN", "RECORDING", "EVALUATING"):
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
        self.recording.start()
        self.buffer_bar.setValue(0)
        self.recording_timer.start(round(self.recording.duration * 1000))
        self.preview_widget.begin_recording(self.recording.duration)
        self.status_pill.setText("🔴 ИДЕТ ЗАПИСЬ ЖЕСТА...")
        self.status_pill.setStyleSheet(
            "background-color: #450a0a; color: #f87171; border: 1.5px solid #ef4444; "
            "border-radius: 8px; padding: 6px 16px; font-size: 15px; font-weight: 800;"
        )
        self.hint_lbl.setText(f"Выполните жест целиком за {self.recording.duration:g} сек")

    def _on_frame_ready(self, frame_bgr: np.ndarray, features_118: np.ndarray, primary_hand: Optional[np.ndarray]) -> None:
        if not self._active:
            return
        if self._camera_failed:
            self._camera_failed = False
            self.btn_action.setEnabled(self._model_ready)
            self.status_pill.setText("Камера подключена")
            self.hint_lbl.setText("Камера подключена. Начните новую запись.")
        if self.state == "RECORDING":
            self.recording.append(features_118)
            self.buffer_bar.setValue(round(self.recording.progress() * 16))

        # 2. Рендеринг кадра с калибровочной рамкой
        frame_with_box, _ = self.ghost_renderer.draw_calibration_box(frame_bgr)
        self.video_canvas.update_frame(frame_with_box)

    def _evaluate_gesture(self) -> None:
        if not self._active or self.state != "RECORDING":
            return
        self.state = "EVALUATING"
        self.preview_widget.finish_recording()
        try:
            raw_16 = self.recording.sample()
        except ValueError as exc:
            self.hint_lbl.setText(str(exc))
            self._reset_state()
            return
        self.buffer_bar.setValue(16)
        self.status_pill.setText("Проверяем жест...")
        self._request_id = self.inference.verify(raw_16, self.target_word)

    def _on_model_loaded(self, request_id):
        if not self._active or request_id != self._request_id:
            return
        self._model_ready = True
        self.btn_action.setEnabled(not self._camera_failed)
        if not self._camera_failed:
            self.hint_lbl.setText(f"Нажмите «Старт» и выполните жест целиком за {self.recording.duration:g} сек")

    def _on_camera_error(self, message):
        if not self._active:
            return
        self._camera_failed = True
        self.preview_widget.finish_recording()
        self.countdown_timer.stop()
        self.recording_timer.stop()
        if self.state == "EVALUATING":
            self._request_id = None
        self.state = "IDLE"
        self.btn_action.setEnabled(False)
        self.status_pill.setText("Камера недоступна")
        self.hint_lbl.setText(f"{message}. Переподключите камеру и повторите запись.")
        self.video_canvas.set_placeholder_text("Камера недоступна")

    def _on_inference_error(self, request_id, message):
        if not self._active or request_id != self._request_id:
            return
        self.hint_lbl.setText(f"Ошибка распознавания: {message}")
        self._reset_state()
        self.btn_action.setEnabled(self._model_ready and not self._camera_failed)

    def _on_verified(self, request_id, res: VerificationResult) -> None:
        if not self._active or request_id != self._request_id or self.state != "EVALUATING":
            return
        self._request_id = None

        if res.is_passed:
            self.state = "SUCCESS"
            self.audio.play_success()

            # Обновление базы данных
            self.db.update_gesture_progress(self.target_word, res.target_confidence, is_passed=True)
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
        self.status_pill.setText("Повторите запись")
        self.btn_action.setEnabled(True)
        self.btn_action.setText("▶ Начать запись (3-2-1)")
