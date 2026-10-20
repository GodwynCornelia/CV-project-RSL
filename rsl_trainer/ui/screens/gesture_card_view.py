from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from rsl_trainer.ui.components.gesture_preview import GestureSilhouettePreview as SkeletonAnimationPreview
from rsl_trainer.core.trajectories import find_trajectory


class GestureCardViewScreen(QWidget):
    """
    Экран Б4. Карточка динамического жеста (по схеме).
    - Интерактивная циклическая визуализация скелета жеста
    - Артикуляция и фазы движения
    - Кнопка перехода в режим тренировки
    """

    def __init__(self, navigator, parent=None, config=None):
        super().__init__(parent)
        self.navigator = navigator
        self.config = config
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
        self.preview_widget = SkeletonAnimationPreview(self, config=self.config)
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
        tips_text = QLabel("Сначала изучите движение на скорости 0,5×. Используйте паузу, стрелки и ползунок для разбора фаз. Скорость показа не меняет время записи попытки.", tips_box)
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

        self.preview_widget.set_trajectory(find_trajectory(word, self.config))

    def on_leave(self) -> None:
        self.preview_widget.stop_animation()

    def _go_back(self) -> None:
        self.navigator.pop_screen()

    def _start_training(self) -> None:
        self.navigator.push_screen("gesture_trainer", target_word=self.current_word)
