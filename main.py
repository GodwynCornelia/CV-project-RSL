import os
import sys

# Обеспечиваем доступность модулей проекта в sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication

from rsl_trainer.ui.main_window import MainWindow


def main():
    # Настройка рендеринга Qt
    app = QApplication(sys.argv)
    app.setApplicationName("Тренажер РЖЯ")
    app.setOrganizationName("RSL-Project")

    # Базовый шрифт системы
    font = QFont("Segoe UI", 10)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    app.setFont(font)

    # Инициализация и запуск главного окна
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
