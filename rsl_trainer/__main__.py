from rsl_trainer.ui.main_window import MainWindow
import sys
from PyQt6.QtWidgets import QApplication

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Тренажер РЖЯ")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
