from pathlib import Path
from typing import Dict
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QFont, QPainter
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget
)

from rsl_trainer.core.dactyl_evaluator import DactylEvaluator


# Справочник артикуляции пальцев для букв РЖЯ
DACTYL_DESCRIPTIONS: Dict[str, str] = {
    "А": "Ладонь сжата в кулак, большой палец плотно прижат сбоку к указательному пальцу.",
    "Б": "Четыре пальца выпрямлены вверх и плотно сомкнуты, большой палец согнут поперек ладони.",
    "В": "Четыре пальца выпрямлены вверх и разведены, большой палец прижат к ребру ладони.",
    "Г": "Указательный палец согнут под углом 90 градусов (крючок), остальные пальцы собраны.",
    "Д": "Указательный палец выпрямлен вертикально вверх, кончики большого, среднего и безымянного пальцев сомкнуты в кольцо.",
    "Е": "Все пальцы слегка согнуты и опущены кончиками к ладони, образуя полукруг.",
    "Ё": "Аналогично букве 'Е', выполняется с легким покачиванием кисти из стороны в сторону.",
    "Ж": "Пальцы разведены и слегка согнуты, имитируя лапки жука.",
    "З": "Указательный палец выпрямлен и чертит в воздухе зигзагообразную цифру '3'.",
    "И": "Мизинец направлен вертикально вверх, остальные пальцы собраны в кулак, удерживаемые большим.",
    "Й": "Мизинец направлен вверх с коротким волнообразным движением кисти.",
    "К": "Указательный и средний пальцы выпрямлены в форме буквы 'V', большой палец лежит между ними.",
    "Л": "Указательный и большой пальцы разведены, образуя угол крыши буквы 'Л'.",
    "М": "Три пальца (указательный, средний, безымянный) направлены вниз, опираясь на большой палец.",
    "Н": "Указательный и средний пальцы направлены вниз, образуя ножки буквы 'Н'.",
    "О": "Все пальцы сомкнуты кончиками с большим пальцем, образуя круглую форму буквы 'О'.",
    "П": "Указательный и средний пальцы согнуты под прямым углом вперед, как перекладина.",
    "Р": "Указательный и средний пальцы скрещены между собой.",
    "С": "Кисть образует полумесяц (букву 'С') полусогнутыми пальцами и большим пальцем.",
    "Т": "Большой палец просунут между указательным и средним пальцами в кулаке.",
    "У": "Большой палец и мизинец отведены в стороны (знак 'рожки'), остальные согнуты.",
    "Ф": "Большой и указательный пальцы образуют кольцо, остальные пальцы выпрямлены вверх.",
    "Х": "Указательный палец согнут крючком, кисть обращена боком.",
    "Ц": "Указательный и средний пальцы направлены вниз, мизинец слегка отведен.",
    "Ч": "Четыре пальца собраны в щепоть и слегка согнуты кончиками вверх.",
    "Ш": "Три средних пальца (указательный, средний, безымянный) выпрямлены вверх и разведены.",
    "Щ": "Три средних пальца направлены вверх, мизинец опущен вниз, как хвостик буквы 'Щ'.",
    "Ъ": "Кисть образует форму буквы 'Т' с коротким горизонтальным движением руки от себя.",
    "Ы": "Кулак обращен ладонью к себе, мизинец поднят вверх.",
    "Ь": "Большой и указательный пальцы полусогнуты, напоминая дугу буквы 'Ь'.",
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

    def set_hand(self, pts_21x2):
        self.hand_pts = pts_21x2
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
        norm = DactylEvaluator.normalize_single_hand(self.hand_pts)
        cx, cy = w // 2, int(h * 0.75)
        scale = min(w, h) * 0.45

        screen_pts = []
        for p in norm:
            sx = int(cx + p[0] * scale)
            sy = int(cy + p[1] * scale)
            screen_pts.append((sx, sy))

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]

        # Линии
        pen = QPen(QColor("#38bdf8"), 3)
        painter.setPen(pen)
        for i1, i2 in connections:
            p1, p2 = screen_pts[i1], screen_pts[i2]
            painter.drawLine(p1[0], p1[1], p2[0], p2[1])

        # Суставы
        for x, y in screen_pts:
            painter.setBrush(QColor("#10b981"))
            painter.setPen(QColor("#ffffff"))
            painter.drawEllipse(x - 4, y - 4, 8, 8)


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
