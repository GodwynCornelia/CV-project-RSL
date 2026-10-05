from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget
)

from rsl_trainer.database.db_manager import DatabaseManager


class MainMenuScreen(QWidget):
    """
    Экран 1. Главное меню («НАЗВ.» по схеме).
    Верхняя панель: Инфо [i], Настройки [⚙], показатели XP и Streak.
    Центральный блок: Две навигационные плитки: «Дактиль» и «Динамические жесты».
    """

    def __init__(self, navigator, db: DatabaseManager, parent=None):
        super().__init__(parent)
        self.navigator = navigator
        self.db = db

        self._build_ui()

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(32, 24, 32, 32)
        main_layout.setSpacing(24)

        # 1. Верхняя панель (Header Bar)
        header = QHBoxLayout()

        self.btn_info = QPushButton("ℹ️", self)
        self.btn_info.setProperty("class", "icon-btn")
        self.btn_info.setToolTip("Справка о РЖЯ и калибровке камеры")
        self.btn_info.clicked.connect(lambda: self.navigator.push_screen("info"))
        header.addWidget(self.btn_info)

        # Название приложения
        title_box = QVBoxLayout()
        app_title = QLabel("ТРЕНАЖЕР РЖЯ", self)
        app_title.setStyleSheet("font-size: 20px; font-weight: 900; color: #38bdf8; letter-spacing: 1.5px;")
        app_sub = QLabel("Интерактивная система изучения жестового языка (Beta MVP)", self)
        app_sub.setStyleSheet("font-size: 12px; color: #94a3b8;")
        title_box.addWidget(app_title)
        title_box.addWidget(app_sub)
        header.addLayout(title_box)

        header.addStretch()

        # Бейджи прогресса (Streak, XP)
        self.badge_streak = QLabel("🔥 1 день", self)
        self.badge_streak.setStyleSheet(
            "background-color: #451a03; color: #fbbf24; border: 1.5px solid #d97706; "
            "border-radius: 8px; padding: 8px 14px; font-weight: bold; font-size: 13px;"
        )
        header.addWidget(self.badge_streak)

        self.badge_xp = QLabel("⚡ 0 XP", self)
        self.badge_xp.setStyleSheet(
            "background-color: #064e3b; color: #34d399; border: 1.5px solid #059669; "
            "border-radius: 8px; padding: 8px 14px; font-weight: bold; font-size: 13px;"
        )
        header.addWidget(self.badge_xp)

        self.btn_settings = QPushButton("⚙️", self)
        self.btn_settings.setProperty("class", "icon-btn")
        self.btn_settings.setToolTip("Настройки камеры и приложения")
        self.btn_settings.clicked.connect(lambda: self.navigator.push_screen("settings"))
        header.addWidget(self.btn_settings)

        main_layout.addLayout(header)

        # Разделитель
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #1e293b;")
        main_layout.addWidget(sep)

        main_layout.addStretch()

        # 2. Центральные навигационные плитки
        tiles_layout = QVBoxLayout()
        tiles_layout.setSpacing(18)

        # Плитка 1: ДАКТИЛЬ
        self.btn_dactyl = QPushButton(self)
        self.btn_dactyl.setProperty("class", "nav-tile")
        self.btn_dactyl.clicked.connect(lambda: self.navigator.push_screen("dactyl_menu"))

        dactyl_inner = QHBoxLayout(self.btn_dactyl)
        dactyl_icon = QLabel("✋", self.btn_dactyl)
        dactyl_icon.setStyleSheet("font-size: 42px; background: transparent;")
        dactyl_inner.addWidget(dactyl_icon)
        dactyl_inner.addSpacing(16)

        dactyl_text = QVBoxLayout()
        dactyl_title = QLabel("ДАКТИЛЬ (РУЧНАЯ АЗБУКА)", self.btn_dactyl)
        dactyl_title.setStyleSheet("font-size: 20px; font-weight: 800; color: #f8fafc; background: transparent;")
        dactyl_desc = QLabel("Изучение 33 букв русского алфавита, постановка пальцев и проверочные тесты", self.btn_dactyl)
        dactyl_desc.setStyleSheet("font-size: 13px; color: #94a3b8; background: transparent;")
        dactyl_text.addWidget(dactyl_title)
        dactyl_text.addWidget(dactyl_desc)
        dactyl_inner.addLayout(dactyl_text)
        dactyl_inner.addStretch()

        arrow1 = QLabel("➔", self.btn_dactyl)
        arrow1.setStyleSheet("font-size: 24px; color: #38bdf8; font-weight: bold; background: transparent;")
        dactyl_inner.addWidget(arrow1)

        tiles_layout.addWidget(self.btn_dactyl)

        # Плитка 2: ДИНАМИЧЕСКИЕ ЖЕСТЫ
        self.btn_dynamic = QPushButton(self)
        self.btn_dynamic.setProperty("class", "nav-tile")
        self.btn_dynamic.clicked.connect(lambda: self.navigator.push_screen("dynamic_menu"))

        dyn_inner = QHBoxLayout(self.btn_dynamic)
        dyn_icon = QLabel("🏃", self.btn_dynamic)
        dyn_icon.setStyleSheet("font-size: 42px; background: transparent;")
        dyn_inner.addWidget(dyn_icon)
        dyn_inner.addSpacing(16)

        dyn_text = QVBoxLayout()
        dyn_title = QLabel("ДИНАМИЧЕСКИЕ ЖЕСТЫ И ФРАЗЫ", self.btn_dynamic)
        dyn_title.setStyleSheet("font-size: 20px; font-weight: 800; color: #f8fafc; background: transparent;")
        dyn_desc = QLabel("Словарь слов, интерактивный тренажер движений и пошаговые уровни фраз", self.btn_dynamic)
        dyn_desc.setStyleSheet("font-size: 13px; color: #94a3b8; background: transparent;")
        dyn_text.addWidget(dyn_title)
        dyn_text.addWidget(dyn_desc)
        dyn_inner.addLayout(dyn_text)
        dyn_inner.addStretch()

        arrow2 = QLabel("➔", self.btn_dynamic)
        arrow2.setStyleSheet("font-size: 24px; color: #10b981; font-weight: bold; background: transparent;")
        dyn_inner.addWidget(arrow2)

        tiles_layout.addWidget(self.btn_dynamic)

        main_layout.addLayout(tiles_layout)
        main_layout.addStretch()

        # Футер
        footer = QLabel("Версия: 1.0.0-beta • Оптимизировано для CPU • Камера 30 FPS", self)
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setStyleSheet("color: #64748b; font-size: 11px;")
        main_layout.addWidget(footer)

    def on_enter(self, **kwargs) -> None:
        """Обновление показателей прогресса при входе на экран."""
        stats = self.db.get_user_stats()
        self.badge_streak.setText(f"🔥 {stats['streak_days']} дн.")
        self.badge_xp.setText(f"⚡ {stats['total_xp']} XP")
