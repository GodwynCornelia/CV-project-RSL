from typing import Dict
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget
)

from rsl_trainer.core.dactyl_evaluator import DactylEvaluator
from rsl_trainer.ui.components.gesture_preview import fit_points, paint_hand_volume


# Справочник артикуляции пальцев для букв РЖЯ
DACTYL_DESCRIPTIONS: Dict[str, str] = {
    "А": "Ладонь сжата в кулак, большой палец плотно прижат сбоку к указательному пальцу.",
    "Б": "Указательный палец поднят, остальные согнуты. Обратите внимание на положение большого пальца у согнутых пальцев.",
    "В": "Четыре пальца выпрямлены вверх и сомкнуты, большой палец отведен в сторону.",
    "Г": "Указательный палец согнут под углом 90 градусов (крючок), остальные пальцы собраны.",
    "Д": "Указательный палец выпрямлен вертикально вверх, кончики большого, среднего и безымянного пальцев сомкнуты в кольцо.",
    "Е": "Все пальцы слегка согнуты и опущены кончиками к ладони, образуя полукруг.",
    "Ё": "Аналогично букве 'Е', выполняется с легким покачиванием кисти из стороны в сторону.",
    "Ж": "Пальцы разведены и слегка согнуты, имитируя лапки жука.",
    "З": "Указательный палец поднят; буква включает круговое движение кисти. Статичная схема показывает конфигурацию пальцев.",
    "И": "Указательный и средний пальцы подняты рядом, остальные согнуты и удерживаются большим пальцем.",
    "Й": "Исходная конфигурация похожа на «И»; буква включает движение кисти. Статичная схема показывает положение пальцев.",
    "К": "Указательный и средний пальцы подняты рядом. Сохраняйте положение большого пальца и согнутых пальцев, показанное на схеме.",
    "Л": "Указательный и средний пальцы направлены вниз и разведены; остальные пальцы согнуты.",
    "М": "Три пальца (указательный, средний, безымянный) направлены вниз, опираясь на большой палец.",
    "Н": "Указательный и средний пальцы направлены вниз, образуя ножки буквы 'Н'.",
    "О": "Все пальцы сомкнуты кончиками с большим пальцем, образуя круглую форму буквы 'О'.",
    "П": "Указательный и средний пальцы направлены вниз и сближены; остальные пальцы согнуты.",
    "Р": "Указательный и средний пальцы скрещены между собой.",
    "С": "Кисть образует полумесяц (букву 'С') полусогнутыми пальцами и большим пальцем.",
    "Т": "Три пальца направлены вниз и сомкнуты; сохраняйте показанный поворот кисти.",
    "У": "Большой палец и мизинец отведены в стороны (знак 'рожки'), остальные согнуты.",
    "Ф": "Большой и указательный пальцы образуют кольцо, остальные пальцы выпрямлены вверх.",
    "Х": "Указательный палец согнут крючком, кисть обращена боком.",
    "Ц": "Указательный и средний пальцы подняты рядом; буква включает короткое движение кисти вниз.",
    "Ч": "Четыре пальца собраны в щепоть и слегка согнуты кончиками вверх.",
    "Ш": "Указательный, средний и безымянный пальцы подняты и сближены; остальные согнуты.",
    "Щ": "Конфигурация пальцев похожа на «Ш»; буква включает движение кисти вниз.",
    "Ъ": "Указательный и большой пальцы разогнуты под углом, остальные согнуты; буква включает движение кисти.",
    "Ы": "Указательный палец и мизинец подняты, средний и безымянный согнуты и удерживаются большим пальцем.",
    "Ь": "Указательный и большой пальцы разогнуты под углом; буква включает движение кисти.",
    "Э": "Полукруглый профиль ладони, направленный ребром вперед.",
    "Ю": "Большой палец и мизинец отведены в стороны, кисть совершает плавное вращение.",
    "Я": "Большой и указательный пальцы выпрямлены, средний палец согнут поперек ладони."
}


class HandSkeletonPreview(QFrame):
    """Виджет отрисовки эталонного каркаса кисти буквы."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(240, 240)
        self.setStyleSheet("background-color: #050811; border: 1.5px solid #1e293b; border-radius: 12px;")
        self.hand_pts = None
        self.yaw = self.pitch = 0.0
        self._drag_start = None
        self.setToolTip("Потяните мышью, чтобы повернуть оболочку; двойной щелчок — исходный вид. Глубина суставов в 2D-данных не измерена.")

    def set_hand(self, pts_21x2):
        self.hand_pts = pts_21x2
        self.yaw = self.pitch = 0.0
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.position()

    def mouseMoveEvent(self, event):
        if self._drag_start is not None and event.buttons() & Qt.MouseButton.LeftButton:
            delta = event.position() - self._drag_start
            self.yaw = float(np.clip(self.yaw + delta.x() * 0.4, -70, 70))
            self.pitch = float(np.clip(self.pitch - delta.y() * 0.4, -60, 60))
            self._drag_start = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        self._drag_start = None

    def mouseDoubleClickEvent(self, event):
        self.yaw = self.pitch = 0.0
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        if self.hand_pts is None or len(self.hand_pts) != 21:
            painter.setPen(QColor("#64748b"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Эталон руки")
            return

        # Центрирование и масштабирование
        fitted, _ = fit_points(self.hand_pts, w, h - 42, padding=28, mirror=True)
        paint_hand_volume(painter, fitted, QColor("#9ce8dd"), self.yaw, self.pitch,
                          viewport=(w, h - 42, 24))
        painter.setPen(QColor("#94a3b8"))
        painter.drawText(self.rect().adjusted(8, h - 42, -8, -4),
                         Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                         "Объёмная схема · потяните для поворота")


class LetterCardScreen(QWidget):
    """
    Экран А3. Карточка буквы (Демонстрация по схеме).
    Отображает эталонную артикуляцию, словесную подсказку и кнопку «Тренировка».
    """

    def __init__(self, navigator, parent=None):
        super().__init__(parent)
        self.navigator = navigator
        self.current_letter = "А"
        self.evaluator = DactylEvaluator()

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(32, 24, 32, 32)
        layout.setSpacing(20)

        # Верхняя панель
        header = QHBoxLayout()
        btn_back = QPushButton("← В алфавит", self)
        btn_back.setProperty("class", "secondary-btn")
        btn_back.clicked.connect(self.navigator.pop_screen)
        header.addWidget(btn_back)

        self.title_lbl = QLabel("КАРТОЧКА БУКВЫ", self)
        self.title_lbl.setStyleSheet("font-size: 18px; font-weight: 800; color: #38bdf8;")
        header.addWidget(self.title_lbl)
        header.addStretch()
        layout.addLayout(header)

        # Центральный блок демонстрации
        content_frame = QFrame()
        content_frame.setProperty("class", "panel")
        c_layout = QHBoxLayout(content_frame)
        c_layout.setContentsMargins(28, 28, 28, 28)
        c_layout.setSpacing(28)

        # Левая часть: Эталонный визуал руки
        self.preview_widget = HandSkeletonPreview(content_frame)
        c_layout.addWidget(self.preview_widget, stretch=1)

        # Правая часть: Описание и подсказки
        info_layout = QVBoxLayout()
        info_layout.setSpacing(14)

        self.letter_hero = QLabel("БУКВА «А»")
        self.letter_hero.setStyleSheet("font-size: 32px; font-weight: 900; color: #f8fafc;")
        info_layout.addWidget(self.letter_hero)

        desc_title = QLabel("Правильная артикуляция пальцев:")
        desc_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #38bdf8;")
        info_layout.addWidget(desc_title)

        self.desc_text = QLabel()
        self.desc_text.setWordWrap(True)
        self.desc_text.setStyleSheet("font-size: 15px; color: #cbd5e1; line-height: 1.5;")
        info_layout.addWidget(self.desc_text)

        tip_box = QFrame()
        tip_box.setStyleSheet("background-color: #0f172a; border-left: 3px solid #10b981; border-radius: 4px; padding: 10px;")
        tip_l = QVBoxLayout(tip_box)
        tip_lbl = QLabel("💡 Совет: В тренажере держите руку прямо перед камерой внутри контура калибровки в течение 1.5 секунд.")
        tip_lbl.setWordWrap(True)
        tip_lbl.setStyleSheet("color: #94a3b8; font-size: 13px;")
        tip_l.addWidget(tip_lbl)
        info_layout.addWidget(tip_box)

        info_layout.addStretch()

        # Кнопка перехода к тренировке с камерой
        self.btn_train = QPushButton("🚀 Начать тренировку с камерой", self)
        self.btn_train.setProperty("class", "primary-btn")
        self.btn_train.clicked.connect(self._go_to_trainer)
        info_layout.addWidget(self.btn_train)

        c_layout.addLayout(info_layout, stretch=2)
        layout.addWidget(content_frame)

    def on_enter(self, letter: str = "А", **kwargs) -> None:
        self.current_letter = letter
        self.title_lbl.setText(f"ОБУЧЕНИЕ: БУКВА «{letter}»")
        self.letter_hero.setText(f"БУКВА «{letter}»")

        # Текстовое описание
        desc = DACTYL_DESCRIPTIONS.get(letter, "Повторите эталонное положение пальцев перед объективом камеры.")
        self.desc_text.setText(desc)

        # Загрузка каркаса руки для превью
        hand_pts = self.evaluator._get_template_hand(letter)
        self.preview_widget.set_hand(hand_pts)

    def set_params(self, **kwargs) -> None:
        """Алиас для установки параметров экрана."""
        self.on_enter(**kwargs)

    def _go_to_trainer(self) -> None:
        self.navigator.push_screen("dactyl_trainer", letter=self.current_letter)
