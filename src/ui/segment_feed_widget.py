from typing import List
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget
)

from src.models import CandidateLattice


class SegmentCard(QFrame):
    """Карточка одного распознанного жестового сегмента с Top-5 распределением."""

    def __init__(self, lattice: CandidateLattice, parent=None):
        super().__init__(parent)
        self.setProperty("class", "card")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        # 1. Шапка карточки: Номер сегмента и таймкод
        header_layout = QHBoxLayout()
        seg_lbl = QLabel(f"ЖЕСТ #{lattice.segment_id}")
        seg_lbl.setStyleSheet("font-weight: 700; color: #38bdf8; font-size: 11px;")
        header_layout.addWidget(seg_lbl)

        header_layout.addStretch()

        ts_lbl = QLabel(f"{lattice.timestamp[0]:.1f}s — {lattice.timestamp[1]:.1f}s")
        ts_lbl.setStyleSheet("color: #64748b; font-size: 10px;")
        header_layout.addWidget(ts_lbl)
        layout.addLayout(header_layout)

        # 2. Победитель (Top-1)
        top1 = lattice.top_hypothesis
        if top1:
            winner_layout = QHBoxLayout()
            word_lbl = QLabel(top1.gloss.upper())
            word_lbl.setStyleSheet("font-size: 14px; font-weight: 700; color: #f8fafc;")
            winner_layout.addWidget(word_lbl)

            winner_layout.addStretch()

            badge_lbl = QLabel(f"{top1.score * 100:.1f}%")
            badge_class = "badge-high" if top1.score >= 0.6 else ("badge-mid" if top1.score >= 0.3 else "badge-low")
            badge_lbl.setProperty("class", badge_class)
            winner_layout.addWidget(badge_lbl)
            layout.addLayout(winner_layout)

        # 3. Список остальных кандидатов (Top-2..Top-5)
        if len(lattice.candidates) > 1:
            sub_layout = QHBoxLayout()
            sub_layout.setSpacing(4)
            for cand in lattice.candidates[1:]:
                sub_badge = QLabel(f"{cand.gloss} ({cand.score * 100:.0f}%)")
                sub_badge.setStyleSheet(
                    "background-color: #0f172a; color: #94a3b8; font-size: 9px; "
                    "border: 1px solid #1e293b; border-radius: 4px; padding: 2px 4px;"
                )
                sub_layout.addWidget(sub_badge)
            sub_layout.addStretch()
            layout.addLayout(sub_layout)


class SegmentFeedWidget(QFrame):
    """Лента сегментов: последовательное накопление карточек жестов во фразе."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("class", "panel")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(8)

        # Заголовок ленты
        title_layout = QHBoxLayout()
        title = QLabel("ЛЕНТА РАСПОЗНАННЫХ ЖЕСТОВ (TOP-5 LATTICE)")
        title.setProperty("class", "heading")
        title_layout.addWidget(title)
        title_layout.addStretch()

        self.count_badge = QLabel("0 жестов")
        self.count_badge.setStyleSheet("color: #94a3b8; font-size: 11px;")
        title_layout.addWidget(self.count_badge)
        main_layout.addLayout(title_layout)

        # Область прокрутки
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("background: transparent; border: none;")

        self.container = QWidget()
        self.container.setStyleSheet("background: transparent;")
        self.cards_layout = QVBoxLayout(self.container)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(8)
        self.cards_layout.addStretch()

        self.scroll_area.setWidget(self.container)
        main_layout.addWidget(self.scroll_area)

        self._cards_count = 0

    def add_segment(self, lattice: CandidateLattice) -> None:
        """Добавляет новую карточку жеста в ленту."""
        card = SegmentCard(lattice)
        # Вставляем перед растягивающим элементом внизу
        self.cards_layout.insertWidget(self.cards_layout.count() - 1, card)
        self._cards_count += 1
        self.count_badge.setText(f"{self._cards_count} жестов")

        # Автопрокрутка вниз
        self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        )

    def clear(self) -> None:
        """Очистить все карточки из ленты."""
        while self.cards_layout.count() > 1:
            item = self.cards_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self._cards_count = 0
        self.count_badge.setText("0 жестов")
