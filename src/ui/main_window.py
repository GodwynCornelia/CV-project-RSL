import sys
import time
from typing import Optional
import cv2
import numpy as np
from PyQt6.QtCore import QThread, QTimer, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget
)

from src.config import UIConfig
from src.models import CandidateLattice, KinematicState, TranslationOutput
from src.pipeline import ContinuousSignTranslationPipeline
from src.ui.oscillograph_widget import KinematicsOscillographWidget
from src.ui.segment_feed_widget import SegmentFeedWidget
from src.ui.styles import DARK_THEME_QSS
from src.ui.tts import TextToSpeechEngine
from src.ui.video_widget import VideoWidget


class CameraThread(QThread):
    """Фоновый поток захвата кадров с веб-камеры с защитой от блокировки GUI."""
    frame_captured = pyqtSignal(np.ndarray)

    def __init__(self, camera_id: int = 0):
        super().__init__()
        self.camera_id = camera_id
        self._running = False

    def run(self):
        self._running = True
        cap = cv2.VideoCapture(self.camera_id)
        while self._running:
            ret, frame = cap.read()
            if ret and frame is not None:
                self.frame_captured.emit(frame)
            time.sleep(0.03)  # ~30 FPS
        cap.release()

    def stop(self):
        self._running = False
        self.wait(1000)


class TranslationWorker(QThread):
    """Фоновый поток семантического декодирования через Qwen LLM."""
    translation_done = pyqtSignal(TranslationOutput)

    def __init__(self, pipeline: ContinuousSignTranslationPipeline):
        super().__init__()
        self.pipeline = pipeline

    def run(self):
        result = self.pipeline.translate_current_phrase()
        self.translation_done.emit(result)


class MainWindow(QMainWindow):
    """
    Главное окно оператора системы непрерывного перевода жестовой речи (Патентный стенд).
    """

    def __init__(self, pipeline: ContinuousSignTranslationPipeline, config: UIConfig = UIConfig()):
        super().__init__()
        self.pipeline = pipeline
        self.config = config
        self.tts = TextToSpeechEngine()

        self.camera_thread: Optional[CameraThread] = None
        self.translation_worker: Optional[TranslationWorker] = None

        self._init_window()
        self._build_ui()
        self._connect_signals()

    def _init_window(self) -> None:
        self.setWindowTitle(self.config.window_title)
        self.resize(self.config.window_width, self.config.window_height)
        self.setStyleSheet(DARK_THEME_QSS)

    def _build_ui(self) -> None:
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(16, 12, 16, 12)
        root_layout.setSpacing(12)

        # 1. Верхний информационный бар
        header_bar = self._create_header_bar()
        root_layout.addLayout(header_bar)

        # 2. Основная рабочая область (сплиттер: слева видео+осциллограф, справа лента+перевод)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(8)

        # Левая панель: Видеохолст + Осциллограф кинематики
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        self.video_widget = VideoWidget(self)
        left_layout.addWidget(self.video_widget, stretch=3)

        self.oscillograph_widget = KinematicsOscillographWidget(parent=self)
        left_layout.addWidget(self.oscillograph_widget, stretch=1)

        splitter.addWidget(left_container)

        # Правая панель: Лента гипотез (Top-5 Lattice) + Студия перевода LLM
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        self.feed_widget = SegmentFeedWidget(self)
        right_layout.addWidget(self.feed_widget, stretch=2)

        translation_panel = self._create_translation_panel()
        right_layout.addWidget(translation_panel, stretch=1)

        splitter.addWidget(right_container)
        splitter.setSizes([840, 560])

        root_layout.addWidget(splitter)

        # Статус-бар
        self.statusBar().showMessage("Система инициализирована. Ожидание запуска камеры или демо-теста.")

    def _create_header_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        title_lbl = QLabel("СИСТЕМА НЕПРЕРЫВНОГО МАШИННОГО ПЕРЕВОДА РЖЯ")
        title_lbl.setStyleSheet("font-size: 16px; font-weight: 800; color: #38bdf8; letter-spacing: 1px;")
        bar.addWidget(title_lbl)

        badge_pat = QLabel("ПАТЕНТНЫЙ СТЕНД v1.0")
        badge_pat.setStyleSheet(
            "background-color: #1e293b; color: #38bdf8; font-size: 11px; font-weight: 700; "
            "border: 1px solid #0284c7; border-radius: 6px; padding: 4px 8px;"
        )
        bar.addWidget(badge_pat)

        bar.addStretch()

        self.status_llm = QLabel("LLM: Qwen2.5 (Ready)")
        self.status_llm.setStyleSheet("color: #10b981; font-weight: 600; font-size: 12px;")
        bar.addWidget(self.status_llm)

        return bar

    def _create_translation_panel(self) -> QFrame:
        panel = QFrame(self)
        panel.setProperty("class", "panel")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        # Заголовок секции
        h_layout = QHBoxLayout()
        title = QLabel("ИТОГОВЫЙ СИНТЕЗ ПРЕДЛОЖЕНИЯ (QWEN LLM)")
        title.setProperty("class", "heading")
        h_layout.addWidget(title)

        h_layout.addStretch()
        self.latency_lbl = QLabel("Задержка: —")
        self.latency_lbl.setStyleSheet("color: #94a3b8; font-size: 11px;")
        h_layout.addWidget(self.latency_lbl)
        layout.addLayout(h_layout)

        # Поле вывода перевода
        self.translation_box = QTextEdit(self)
        self.translation_box.setProperty("class", "translation-box")
        self.translation_box.setPlaceholderText("Здесь появится связный перевод жестовой фразы...")
        layout.addWidget(self.translation_box)

        # Панель кнопок управления
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)

        self.btn_camera = QPushButton("📹 Камера", self)
        self.btn_camera.setProperty("class", "secondary-btn")
        self.btn_camera.clicked.connect(self._toggle_camera)
        btn_layout.addWidget(self.btn_camera)

        self.btn_test_demo = QPushButton("🧪 Запустить Демо", self)
        self.btn_test_demo.setProperty("class", "secondary-btn")
        self.btn_test_demo.clicked.connect(self._run_demo_phrase)
        btn_layout.addWidget(self.btn_test_demo)

        self.btn_clear = QPushButton("🗑 Очистить", self)
        self.btn_clear.setProperty("class", "secondary-btn")
        self.btn_clear.clicked.connect(self._clear_session)
        btn_layout.addWidget(self.btn_clear)

        btn_layout.addStretch()

        self.btn_speak = QPushButton("🔊 Озвучить", self)
        self.btn_speak.setProperty("class", "secondary-btn")
        self.btn_speak.clicked.connect(self._speak_translation)
        btn_layout.addWidget(self.btn_speak)

        self.btn_translate = QPushButton("⚡ Перевести фразу", self)
        self.btn_translate.setProperty("class", "accent-btn")
        self.btn_translate.clicked.connect(self._trigger_translation)
        btn_layout.addWidget(self.btn_translate)

        layout.addLayout(btn_layout)

        return panel

    def _connect_signals(self) -> None:
        # Привязка коллбэков пайплайна
        self.pipeline.on_kinematics_updated = self._handle_kinematics_update
        self.pipeline.on_lattice_generated = self._handle_new_lattice
        self.pipeline.on_translation_ready = self._handle_translation_ready

    def _handle_kinematics_update(self, state: KinematicState) -> None:
        self.oscillograph_widget.update_kinematics(state)

    def _handle_new_lattice(self, lattice: CandidateLattice) -> None:
        self.feed_widget.add_segment(lattice)
        top1 = lattice.top_hypothesis.gloss if lattice.top_hypothesis else "—"
        self.statusBar().showMessage(f"Распознан жест #{lattice.segment_id}: «{top1}» (Top-1)")

    def _handle_translation_ready(self, output: TranslationOutput) -> None:
        self.translation_box.setText(output.natural_sentence)
        self.latency_lbl.setText(f"LLM: {output.latency_ms:.0f} мс")
        self.statusBar().showMessage("Семантический перевод сформирован!")

    def _toggle_camera(self) -> None:
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
            self.camera_thread = None
            self.btn_camera.setText("📹 Камера")
            self.statusBar().showMessage("Камера остановлена.")
        else:
            self.camera_thread = CameraThread(self.config.camera_id)
            self.camera_thread.frame_captured.connect(self._process_camera_frame)
            self.camera_thread.start()
            self.btn_camera.setText("⏹ Стоп камера")
            self.statusBar().showMessage("Камера активна. Выполняйте жесты перед объективом.")

    def _process_camera_frame(self, frame: np.ndarray) -> None:
        overlay, _, _ = self.pipeline.process_frame(frame)
        self.video_widget.update_frame(overlay)

    def _trigger_translation(self) -> None:
        if not self.pipeline.recognized_lattices:
            self.statusBar().showMessage("Нет распознанных жестов для перевода.")
            return

        self.statusBar().showMessage("Генерация перевода через Qwen LLM...")
        self.translation_worker = TranslationWorker(self.pipeline)
        self.translation_worker.translation_done.connect(self._handle_translation_ready)
        self.translation_worker.start()

    def _speak_translation(self) -> None:
        text = self.translation_box.toPlainText().strip()
        if text:
            self.tts.speak(text)
            self.statusBar().showMessage("Озвучивание текста...")

    def _clear_session(self) -> None:
        self.pipeline.clear_phrase()
        self.feed_widget.clear()
        self.translation_box.clear()
        self.latency_lbl.setText("Задержка: —")
        self.statusBar().showMessage("Фраза очищена.")

    def _run_demo_phrase(self) -> None:
        """
        Запуск встроенной демонстрации для патентной комиссии/тестирования:
        Синтезирует последовательность жестов из локального репозитория,
        прогоняет через нейросеть и семантический декодер.
        """
        from tests.test_synthetic_e2e import generate_demo_lattices
        self._clear_session()
        self.statusBar().showMessage("Запуск демонстрационного теста...")

        lattices = generate_demo_lattices(self.pipeline.classifier)
        for lat in lattices:
            self.pipeline.recognized_lattices.append(lat)
            self.feed_widget.add_segment(lat)

        self._trigger_translation()

    def closeEvent(self, event):
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
        event.accept()
