import sys
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget
)

from rsl_trainer.config import AppConfig
from rsl_trainer.cv.camera_worker import CameraWorker
from rsl_trainer.database.db_manager import DatabaseManager
from rsl_trainer.ui.navigation import NavigationManager
from rsl_trainer.ui.screens import (
    AlphabetGridScreen,
    DactylMenuScreen,
    DactylTrainerScreen,
    DynamicMenuScreen,
    ExamCameraScreen,
    GestureCardViewScreen,
    GestureTrainerScreen,
    InfoModalScreen,
    LetterCardViewScreen,
    MainMenuScreen,
    SentenceLevelsScreen,
    SettingsModalScreen,
    TestConfigScreen,
    WordCatalogScreen
)
from rsl_trainer.ui.theme import DARK_THEME_QSS


class MainWindow(QMainWindow):
    """
    Главное окно приложения «Тренажер РЖЯ» (Beta MVP).
    - Координирует MVP NavigationManager и экраны приложения
    - Изолирует фоновый CameraWorker (QThread)
    - Обеспечивает отказоустойчивость при сбоях оборудования (Hardware Resilience)
    """

    def __init__(self, config: AppConfig = None, parent=None):
        super().__init__(parent)
        self.config = config or AppConfig()
        self.db = DatabaseManager()

        self.setWindowTitle("Тренажер РЖЯ — Интерактивное обучение жестовому языку (Beta MVP)")
        self.setMinimumSize(1024, 720)
        self.resize(1280, 820)
        self.setStyleSheet(DARK_THEME_QSS)

        # Фоновый воркер камеры
        self.camera_worker = CameraWorker(self.config)
        self.camera_worker.camera_error.connect(self._on_camera_error)

        self._build_ui()
        self._register_screens()

    def _build_ui(self) -> None:
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        self.main_layout = QVBoxLayout(central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Плашка аппаратной ошибки камеры (Hardware Resilience по спецификации 5.2)
        self.error_banner = QFrame(self)
        self.error_banner.setStyleSheet(
            "background-color: #7f1d1d; border-bottom: 2px solid #ef4444; padding: 10px 24px;"
        )
        eb_layout = QHBoxLayout(self.error_banner)
        eb_layout.setContentsMargins(16, 6, 16, 6)

        eb_icon = QLabel("⚠️", self.error_banner)
        eb_icon.setStyleSheet("font-size: 20px; background: transparent;")
        eb_layout.addWidget(eb_icon)

        self.error_lbl = QLabel(
            "Веб-камера отключена или недоступна. Подключите устройство и нажмите «Переподключить».",
            self.error_banner
        )
        self.error_lbl.setStyleSheet("color: #fef2f2; font-size: 13px; font-weight: bold; background: transparent;")
        eb_layout.addWidget(self.error_lbl)
        eb_layout.addStretch()

        btn_reconnect = QPushButton("🔄 Переподключить", self.error_banner)
        btn_reconnect.setStyleSheet(
            "background-color: #dc2626; color: white; font-weight: bold; padding: 6px 14px; "
            "border-radius: 6px; border: 1px solid #f87171;"
        )
        btn_reconnect.clicked.connect(self._retry_camera)
        eb_layout.addWidget(btn_reconnect)

        self.error_banner.hide()
        self.main_layout.addWidget(self.error_banner)

        # Навигатор экранов (QStackedWidget)
        self.navigator = NavigationManager(self)
        self.main_layout.addWidget(self.navigator, stretch=1)

    def _register_screens(self) -> None:
        """Регистрация всех 14 экранов приложения строго по схеме."""
        nav = self.navigator

        nav.register_screen("main_menu", MainMenuScreen(nav, self.db))

        info_scr = InfoModalScreen(nav)
        nav.register_screen("info", info_scr)
        nav.register_screen("info_modal", info_scr)

        settings_scr = SettingsModalScreen(nav, self.config)
        nav.register_screen("settings", settings_scr)
        nav.register_screen("settings_modal", settings_scr)

        # Ветка А: Дактиль
        nav.register_screen("dactyl_menu", DactylMenuScreen(nav))
        nav.register_screen("alphabet_grid", AlphabetGridScreen(nav, self.db))
        letter_card_scr = LetterCardViewScreen(nav)
        nav.register_screen("letter_card", letter_card_scr)
        nav.register_screen("letter_card_view", letter_card_scr)
        nav.register_screen("dactyl_trainer", DactylTrainerScreen(nav, self.camera_worker, self.db))
        nav.register_screen("test_config", TestConfigScreen(nav))
        nav.register_screen("exam_camera", ExamCameraScreen(nav, self.camera_worker, self.db))

        # Ветка Б: Динамические жесты
        nav.register_screen("dynamic_menu", DynamicMenuScreen(nav))
        catalog_scr = WordCatalogScreen(nav, self.db)
        nav.register_screen("word_catalog", catalog_scr)
        nav.register_screen("gesture_catalog", catalog_scr)
        gesture_card_scr = GestureCardViewScreen(nav)
        nav.register_screen("gesture_card", gesture_card_scr)
        nav.register_screen("gesture_card_view", gesture_card_scr)
        nav.register_screen("gesture_trainer", GestureTrainerScreen(nav, self.camera_worker, self.db))
        nav.register_screen("sentence_levels", SentenceLevelsScreen(nav, self.camera_worker, self.db))

        # Стартовый экран
        nav.push_screen("main_menu")

    def _on_camera_error(self, err_msg: str) -> None:
        """Обработка физического отключения камеры или сбоя драйвера."""
        self.error_lbl.setText(f"Сбой камеры: {err_msg}. Подключите камеру и нажмите «Переподключить».")
        self.error_banner.show()

    def _retry_camera(self) -> None:
        """Попытка переподключения камеры."""
        self.error_banner.hide()
        self.camera_worker.stop()
        self.camera_worker.wait(500)
        self.camera_worker.start()

    def closeEvent(self, event) -> None:
        """Безопасная остановка потоков при закрытии приложения."""
        try:
            self.camera_worker.stop()
            self.camera_worker.wait(1500)
        except Exception:
            pass
        event.accept()
