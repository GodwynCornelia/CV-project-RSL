import random
import time
from typing import List, Optional
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QStackedWidget,
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


class ExamCameraScreen(QWidget):
    """
    Экран А6. Камера-экзамен (по схеме).
    - Камера без подсказывающего оверлея
    - Таймер 5 секунд на попытку
    - Случайный порядок заданий
    - Автоматическая фиксация баллов и финальный экран статистики
    """

    def __init__(self, navigator, camera_worker: CameraWorker, db: DatabaseManager, parent=None):
        super().__init__(parent)
        self.navigator = navigator
        self.camera_worker = camera_worker
        self.db = db
        self.audio = AudioFeedback()

        self.evaluator = DactylEvaluator(threshold=0.70, hold_duration=1.0)
        self.ghost_renderer = GhostOverlayRenderer()

        # Параметры экзамена
        self.exam_letters: List[str] = []
        self.current_idx = 0
        self.correct_count = 0
        self.failed_letters: List[str] = []
        self.start_time = 0.0

        # Таймер 5 секунд
        self.seconds_left = 5.0
        self.timer = QTimer(self)
        self.timer.setInterval(100)  # 100 мс тик
        self.timer.timeout.connect(self._on_timer_tick)

        self._build_ui()

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.stack = QStackedWidget(self)

        # Страница 1: Активный экзамен
        self.exam_widget = QWidget(self)
        self._build_exam_page(self.exam_widget)
        self.stack.addWidget(self.exam_widget)

        # Страница 2: Результаты
        self.results_widget = QWidget(self)
        self._build_results_page(self.results_widget)
        self.stack.addWidget(self.results_widget)

        main_layout.addWidget(self.stack)

    def _build_exam_page(self, page: QWidget) -> None:
        layout = QVBoxLayout(page)
        layout.setContentsMargins(28, 16, 28, 20)
        layout.setSpacing(12)

        # Верхняя панель
        header = QHBoxLayout()
        btn_back = QPushButton("✕ Прервать экзамен", page)
        btn_back.setProperty("class", "secondary-btn")
        btn_back.clicked.connect(self._abort_exam)
        header.addWidget(btn_back)

        self.progress_lbl = QLabel("Вопрос 1 из 5", page)
        self.progress_lbl.setStyleSheet("font-size: 16px; font-weight: bold; color: #94a3b8;")
        header.addWidget(self.progress_lbl)

        header.addStretch()

        # Бейдж таймера
        self.timer_badge = QLabel("⏱ 5.0с", page)
        self.timer_badge.setStyleSheet(
            "background-color: #1e293b; color: #38bdf8; border: 1.5px solid #0284c7; "
            "border-radius: 8px; padding: 6px 16px; font-size: 16px; font-weight: 800;"
        )
        header.addWidget(self.timer_badge)
        layout.addLayout(header)

        # Плашка задания
        task_box = QFrame(page)
        task_box.setStyleSheet(
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1e293b, stop:1 #0f172a); "
            "border: 1.5px solid #334155; border-radius: 12px; padding: 8px 16px;"
        )
        task_layout = QHBoxLayout(task_box)
        task_title = QLabel("ПОКАЖИТЕ БУКВУ:", task_box)
        task_title.setStyleSheet("font-size: 16px; color: #94a3b8; font-weight: bold;")
        task_layout.addWidget(task_title)

        self.target_letter_lbl = QLabel("А", task_box)
        self.target_letter_lbl.setStyleSheet(
            "font-size: 32px; font-weight: 900; color: #f59e0b; "
            "padding: 0 12px; background: transparent;"
        )
        task_layout.addWidget(self.target_letter_lbl)
        task_layout.addStretch()

        self.task_hint = QLabel("Удерживайте руку в заданной позе", task_box)
        self.task_hint.setStyleSheet("color: #64748b; font-size: 13px;")
        task_layout.addWidget(self.task_hint)
        layout.addWidget(task_box)

        # Холст видео
        self.video_canvas = VideoCanvas(page)
        layout.addWidget(self.video_canvas, stretch=1)

    def _build_results_page(self, page: QWidget) -> None:
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 32, 40, 32)
        layout.setSpacing(24)

        layout.addStretch()

        container = QFrame(page)
        container.setStyleSheet(
            "background-color: #1e293b; border: 2px solid #334155; border-radius: 20px; padding: 32px;"
        )
        c_layout = QVBoxLayout(container)
        c_layout.setSpacing(18)

        self.res_icon = QLabel("🏆", container)
        self.res_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.res_icon.setStyleSheet("font-size: 54px; background: transparent;")
        c_layout.addWidget(self.res_icon)

        self.res_title = QLabel("ЭКЗАМЕН ЗАВЕРШЕН!", container)
        self.res_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.res_title.setStyleSheet("font-size: 24px; font-weight: 900; color: #f8fafc;")
        c_layout.addWidget(self.res_title)

        self.res_score = QLabel("Результат: 5 из 5 (100%)", container)
        self.res_score.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.res_score.setStyleSheet("font-size: 20px; font-weight: 800; color: #34d399;")
        c_layout.addWidget(self.res_score)

        self.res_xp = QLabel("+50 XP начислено!", container)
        self.res_xp.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.res_xp.setStyleSheet("font-size: 15px; font-weight: bold; color: #fbbf24;")
        c_layout.addWidget(self.res_xp)

        self.res_time = QLabel("Затраченное время: 24.5 с", container)
        self.res_time.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.res_time.setStyleSheet("font-size: 14px; color: #94a3b8;")
        c_layout.addWidget(self.res_time)

        self.res_errors = QLabel("Ошибки в буквах: нет", container)
        self.res_errors.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.res_errors.setStyleSheet("font-size: 14px; color: #f87171; font-weight: bold;")
        c_layout.addWidget(self.res_errors)

        c_layout.addSpacing(16)

        btn_row = QHBoxLayout()
        btn_retry = QPushButton("🔄 Пройти еще раз", container)
        btn_retry.setProperty("class", "secondary-btn")
        btn_retry.clicked.connect(self._restart_exam)
        btn_row.addWidget(btn_retry)

        btn_finish = QPushButton("В меню дактиля ➔", container)
        btn_finish.setProperty("class", "primary-btn")
        btn_finish.clicked.connect(self._finish_and_leave)
        btn_row.addWidget(btn_finish)
        c_layout.addLayout(btn_row)

        layout.addWidget(container)
        layout.addStretch()

    def on_enter(self, question_count: int = 5, **kwargs) -> None:
        """Инициализация экзаменационной сессии."""
        self.question_count = min(question_count, len(RUSSIAN_ALPHABET))
        # Выбираем случайные уникальные буквы
        self.exam_letters = random.sample(RUSSIAN_ALPHABET, self.question_count)
        self.current_idx = 0
        self.correct_count = 0
        self.failed_letters = []
        self.start_time = time.time()

        self.stack.setCurrentIndex(0)

        # Переключаем трекер в режим DACTYL (быстрый MediaPipe)
        self.camera_worker.set_mode(TrackerMode.DACTYL)
        self.camera_worker.frame_ready.connect(self._on_frame_ready)

        if not self.camera_worker.isRunning():
            self.camera_worker.start()

        self._setup_question()

    def on_leave(self) -> None:
        self.timer.stop()
        try:
            self.camera_worker.frame_ready.disconnect(self._on_frame_ready)
        except Exception:
            pass

    def _abort_exam(self) -> None:
        self.on_leave()
        self.navigator.pop_screen()

    def _setup_question(self) -> None:
        """Подготовка следующей буквы."""
        if self.current_idx >= len(self.exam_letters):
            self._show_results()
            return

        current_letter = self.exam_letters[self.current_idx]
        self.target_letter_lbl.setText(current_letter)
        self.progress_lbl.setText(f"Вопрос {self.current_idx + 1} из {len(self.exam_letters)}")

        self.evaluator.set_target_letter(current_letter)
        self.seconds_left = 5.0
        self.timer_badge.setText("⏱ 5.0с")
        self.timer_badge.setStyleSheet(
            "background-color: #1e293b; color: #38bdf8; border: 1.5px solid #0284c7; "
            "border-radius: 8px; padding: 6px 16px; font-size: 16px; font-weight: 800;"
        )
        self.timer.start()

    def _on_timer_tick(self) -> None:
        self.seconds_left -= 0.1
        if self.seconds_left <= 0:
            self.timer.stop()
            self._on_timeout()
            return

        # Обновление бейджа таймера
        self.timer_badge.setText(f"⏱ {max(0.0, self.seconds_left):.1f}с")
        if self.seconds_left <= 2.0:
            self.timer_badge.setStyleSheet(
                "background-color: #450a0a; color: #f87171; border: 1.5px solid #ef4444; "
                "border-radius: 8px; padding: 6px 16px; font-size: 16px; font-weight: 800;"
            )
        elif self.seconds_left <= 3.5:
            self.timer_badge.setStyleSheet(
                "background-color: #451a03; color: #fbbf24; border: 1.5px solid #d97706; "
                "border-radius: 8px; padding: 6px 16px; font-size: 16px; font-weight: 800;"
            )

    def _on_timeout(self) -> None:
        """Время истекло без сдачи буквы."""
        curr_letter = self.exam_letters[self.current_idx]
        self.failed_letters.append(curr_letter)
        self.audio.play_error()

        self.current_idx += 1
        self._setup_question()

    def _on_frame_ready(self, frame_bgr: np.ndarray, features_118: np.ndarray, primary_hand: Optional[np.ndarray]) -> None:
        if self.stack.currentIndex() != 0:
            return

        # В экзамене рисуем только калибровочную рамку (БЕЗ ghost overlay!)
        frame_with_box, calib_box = self.ghost_renderer.draw_calibration_box(frame_bgr)

        # Оценка совпадения
        if primary_hand is not None:
            score, hold_prog, is_confirmed = self.evaluator.evaluate_hand(primary_hand)
        else:
            score, hold_prog, is_confirmed = 0.0, 0.0, False

        self.video_canvas.set_hold_metrics(score, hold_prog, show_bar=True)
        self.video_canvas.update_frame(frame_with_box)

        # Если буква подтверждена до окончания таймера
        if is_confirmed:
            self.timer.stop()
            self.correct_count += 1
            curr_letter = self.exam_letters[self.current_idx]
            self.db.update_dactyl_progress(curr_letter, score)
            self.audio.play_success()

            self.current_idx += 1
            self._setup_question()

    def _show_results(self) -> None:
        """Показ экрана итогов экзамена."""
        self.on_leave()

        elapsed_time = time.time() - self.start_time
        total_questions = len(self.exam_letters)
        percent = (self.correct_count / total_questions) * 100 if total_questions > 0 else 0
        earned_xp = self.correct_count * 10

        self.db.add_xp(earned_xp)
        self.db.update_streak()

        if percent >= 80:
            self.res_icon.setText("🏆")
            self.res_score.setStyleSheet("font-size: 20px; font-weight: 800; color: #34d399;")
        elif percent >= 50:
            self.res_icon.setText("⭐")
            self.res_score.setStyleSheet("font-size: 20px; font-weight: 800; color: #fbbf24;")
        else:
            self.res_icon.setText("💪")
            self.res_score.setStyleSheet("font-size: 20px; font-weight: 800; color: #f87171;")

        self.res_score.setText(f"Результат: {self.correct_count} из {total_questions} ({percent:.0f}%)")
        self.res_xp.setText(f"+{earned_xp} XP начислено!")
        self.res_time.setText(f"Затраченное время: {elapsed_time:.1f} сек")

        if self.failed_letters:
            self.res_errors.setText(f"Ошибки в буквах: {', '.join(self.failed_letters)}")
            self.res_errors.setStyleSheet("font-size: 14px; color: #f87171; font-weight: bold;")
        else:
            self.res_errors.setText("Идеально! Ошибок нет 🎯")
            self.res_errors.setStyleSheet("font-size: 14px; color: #34d399; font-weight: bold;")

        self.stack.setCurrentIndex(1)
        self.audio.play_level_complete()

    def _restart_exam(self) -> None:
        self.on_enter(question_count=len(self.exam_letters))

    def _finish_and_leave(self) -> None:
        self.navigator.pop_screen()
