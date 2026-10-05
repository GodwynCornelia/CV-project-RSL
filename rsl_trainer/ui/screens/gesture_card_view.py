from pathlib import Path
from typing import Optional
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget
)

from rsl_trainer.config import SKELETONS_DIR
from rsl_trainer.cv.ghost_overlay import HAND_CONNECTIONS, POSE_CONNECTIONS


class SkeletonAnimationPreview(QWidget):
    """
    Виджет анимированного предпросмотра эталонной траектории жеста (16 кадров x 118 точек).
    Воспроизводит циклически скелет движения диктора с плавным рендерингом на QPainter.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(420, 320)
        self.setStyleSheet("background-color: #020617; border-radius: 14px; border: 1.5px solid #1e293b;")

        self.trajectory: Optional[np.ndarray] = None  # (16, 118)
        self.frame_idx = 0

        self.anim_timer = QTimer(self)
        self.anim_timer.setInterval(120)  # ~8 FPS для наглядного пошагового движения
        self.anim_timer.timeout.connect(self._on_anim_step)

    def set_trajectory(self, npy_path: Path) -> None:
        if npy_path.exists():
            try:
                arr = np.load(npy_path)
                if arr.shape[0] == 16 and arr.shape[1] >= 118:
                    self.trajectory = arr
                    self.frame_idx = 0
                    self.anim_timer.start()
                    self.update()
                    return
            except Exception:
                pass
        self.trajectory = None
        self.anim_timer.stop()
        self.update()

    def stop_animation(self) -> None:
        self.anim_timer.stop()

    def _on_anim_step(self) -> None:
        if self.trajectory is not None:
            self.frame_idx = (self.frame_idx + 1) % len(self.trajectory)
            self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # Фон
        painter.fillRect(0, 0, w, h, QColor("#020617"))

        if self.trajectory is None:
            painter.setPen(QColor("#64748b"))
            painter.setFont(QFont("Segoe UI", 13))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Эталонная траектория движения формируется..."
            )
            return

        # Извлекаем текущий кадр (118,)
        frame_feats = self.trajectory[self.frame_idx]

        # Первые 34 = тело (17 точек x 2)
        body_pts = frame_feats[:34].reshape(17, 2)
        # Следующие 42 = левая рука (21 x 2)
        lh_pts = frame_feats[34:76].reshape(21, 2)
        # Следующие 42 = правая рука (21 x 2)
        rh_pts = frame_feats[76:118].reshape(21, 2)

        # Центрирование и масштабирование под холст
        scale = min(w, h) * 0.75
        cx = w / 2.0
        cy = h / 2.0

        def to_screen(pt):
            # Нормализованные координаты вокруг (0,0) -> в экранные
            return int(cx + pt[0] * scale), int(cy + pt[1] * scale)

        # 1. Отрисовка скелета тела (синий/фиолетовый тон)
        pen_body = QPen(QColor("#818cf8"), 3)
        painter.setPen(pen_body)
        for i1, i2 in POSE_CONNECTIONS:
            p1 = to_screen(body_pts[i1])
            p2 = to_screen(body_pts[i2])
            painter.drawLine(p1[0], p1[1], p2[0], p2[1])

        painter.setBrush(QColor("#c7d2fe"))
        for i in range(17):
            pt = to_screen(body_pts[i])
            painter.drawEllipse(pt[0] - 3, pt[1] - 3, 6, 6)

        # 2. Отрисовка рук (изумрудный/аква тон)
        pen_hand = QPen(QColor("#34d399"), 2)
        painter.setPen(pen_hand)
        for hand_pts, is_present in [(lh_pts, np.any(lh_pts != 0)), (rh_pts, np.any(rh_pts != 0))]:
            if not is_present:
                continue
            for i1, i2 in HAND_CONNECTIONS:
                p1 = to_screen(hand_pts[i1])
                p2 = to_screen(hand_pts[i2])
                painter.drawLine(p1[0], p1[1], p2[0], p2[1])

            painter.setBrush(QColor("#a7f3d0"))
            for i in range(21):
                pt = to_screen(hand_pts[i])
                painter.drawEllipse(pt[0] - 2, pt[1] - 2, 4, 4)

        # Индикатор шага анимации
        painter.setPen(QColor("#94a3b8"))
        painter.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        painter.drawText(20, h - 20, f"Фаза движения: кадр {self.frame_idx + 1} / {len(self.trajectory)}")


class GestureCardViewScreen(QWidget):
    """
    Экран Б4. Карточка динамического жеста (по схеме).
    - Интерактивная циклическая визуализация скелета жеста
    - Артикуляция и фазы движения
    - Кнопка перехода в режим тренировки
    """

    def __init__(self, navigator, parent=None):
        super().__init__(parent)
        self.navigator = navigator
        self.current_word = "Привет!"
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 20, 32, 24)
        layout.setSpacing(16)

        # Верхняя панель
        header = QHBoxLayout()
        btn_back = QPushButton("← В словарь", self)
        btn_back.setProperty("class", "secondary-btn")
        btn_back.clicked.connect(self._go_back)
        header.addWidget(btn_back)

        self.title_lbl = QLabel("ЖЕСТ: «ПРИВЕТ!»", self)
        self.title_lbl.setStyleSheet("font-size: 20px; font-weight: 800; color: #a855f7;")
        header.addWidget(self.title_lbl)
        header.addStretch()
        layout.addLayout(header)

        # Контентная область: слева аниматор, справа описание
        content_layout = QHBoxLayout()
        content_layout.setSpacing(24)

        # Слева: Аниматор траектории
        self.preview_widget = SkeletonAnimationPreview(self)
        content_layout.addWidget(self.preview_widget, stretch=1)

        # Справа: Информация и подсказка
        info_panel = QFrame(self)
        info_panel.setStyleSheet(
            "background-color: #1e293b; border: 1.5px solid #334155; border-radius: 16px; padding: 24px;"
        )
        info_layout = QVBoxLayout(info_panel)
        info_layout.setSpacing(16)

        desc_head = QLabel("📋 Описание и артикуляция", info_panel)
        desc_head.setStyleSheet("font-size: 16px; font-weight: 800; color: #f8fafc;")
        info_layout.addWidget(desc_head)

        self.desc_body = QLabel(info_panel)
        self.desc_body.setWordWrap(True)
        self.desc_body.setStyleSheet("font-size: 14px; color: #cbd5e1; line-height: 1.5;")
        info_layout.addWidget(self.desc_body)

        tips_box = QFrame(info_panel)
        tips_box.setStyleSheet(
            "background-color: #0f172a; border-left: 3px solid #a855f7; border-radius: 6px; padding: 12px;"
        )
        tips_layout = QVBoxLayout(tips_box)
        tips_title = QLabel("💡 Совет по тренировке:", tips_box)
        tips_title.setStyleSheet("font-size: 13px; font-weight: bold; color: #c084fc;")
        tips_text = QLabel("Выполняйте движение плавно. Запись активируется по кнопке «Старт» и фиксирует окно из 16 кадров.", tips_box)
        tips_text.setStyleSheet("font-size: 12px; color: #94a3b8;")
        tips_text.setWordWrap(True)
        tips_layout.addWidget(tips_title)
        tips_layout.addWidget(tips_text)
        info_layout.addWidget(tips_box)

        info_layout.addStretch()

        self.btn_train = QPushButton("Начать тренировку ➔", info_panel)
        self.btn_train.setProperty("class", "primary-btn")
        self.btn_train.clicked.connect(self._start_training)
        info_layout.addWidget(self.btn_train)

        content_layout.addWidget(info_panel, stretch=1)
        layout.addLayout(content_layout, stretch=1)

    def on_enter(self, word: str = "Привет!", **kwargs) -> None:
        self.current_word = word
        self.title_lbl.setText(f"ЖЕСТ: «{word.upper()}»")

        # Описание жеста
        self.desc_body.setText(
            f"Динамический жест «{word}» в русском жестовом языке.\n\n"
            "Жест выполняется плавным движением руки и кисти в пространстве перед грудью. "
            "Внимательно следите за циклической траекторией суставов на анимации слева: "
            "важно соблюдать правильную конфигурацию пальцев и траекторию смещения кисти."
        )

        # Поиск npy файла
        npy_path = SKELETONS_DIR / f"{word}.npy"
        if not npy_path.exists():
            # Попробуем с маленькой буквы
            npy_path = SKELETONS_DIR / f"{word.lower()}.npy"

        self.preview_widget.set_trajectory(npy_path)

    def on_leave(self) -> None:
        self.preview_widget.stop_animation()

    def _go_back(self) -> None:
        self.on_leave()
        self.navigator.pop_screen()

    def _start_training(self) -> None:
        self.on_leave()
        self.navigator.push_screen("gesture_trainer", target_word=self.current_word)
