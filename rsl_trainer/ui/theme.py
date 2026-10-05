"""
Дизайн-система «Тренажер РЖЯ» (PyQt6).
Содержит цветовые токены, стили и шрифты, удовлетворяющие спецификации доступности.
"""

DARK_THEME = """
QMainWindow, QDialog {
    background-color: #0b0f19;
    color: #f8fafc;
}

QWidget {
    font-family: 'Segoe UI', -apple-system, Roboto, sans-serif;
    color: #e2e8f0;
    font-size: 14px;
}

/* Верхняя навигационная панель */
QFrame#headerBar {
    background-color: #131b2e;
    border-bottom: 1.5px solid #1e293b;
    padding: 8px 16px;
}

/* Информационные плашки и карточки */
QFrame.panel {
    background-color: #131b2e;
    border: 1px solid #1e293b;
    border-radius: 12px;
}

QFrame.card {
    background-color: #182238;
    border: 1.5px solid #29354f;
    border-radius: 12px;
}

QFrame.card:hover {
    border-color: #38bdf8;
    background-color: #1e2b47;
}

/* Главные навигационные плитки */
QPushButton.nav-tile {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1e293b, stop:1 #0f172a);
    border: 2px solid #334155;
    border-radius: 16px;
    padding: 24px;
    text-align: left;
    min-height: 120px;
}

QPushButton.nav-tile:hover {
    border-color: #38bdf8;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1e3a5f, stop:1 #0f172a);
}

QPushButton.nav-tile:pressed {
    background-color: #0c4a6e;
}

/* Кнопки управления (Минимум 48x48 px по спецификации) */
QPushButton.primary-btn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284c7, stop:1 #0369a1);
    color: #ffffff;
    font-size: 15px;
    font-weight: 700;
    border-radius: 10px;
    padding: 12px 24px;
    min-height: 48px;
    border: none;
}

QPushButton.primary-btn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #38bdf8, stop:1 #0284c7);
}

QPushButton.primary-btn:pressed {
    background-color: #075985;
}

QPushButton.secondary-btn {
    background-color: #1e293b;
    color: #e2e8f0;
    font-size: 14px;
    font-weight: 600;
    border-radius: 10px;
    padding: 12px 20px;
    min-height: 48px;
    border: 1.5px solid #334155;
}

QPushButton.secondary-btn:hover {
    background-color: #334155;
    border-color: #475569;
    color: #ffffff;
}

QPushButton.accent-btn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #059669, stop:1 #047857);
    color: #ffffff;
    font-size: 15px;
    font-weight: 700;
    border-radius: 10px;
    padding: 12px 24px;
    min-height: 48px;
    border: none;
}

QPushButton.accent-btn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #10b981, stop:1 #059669);
}

QPushButton.icon-btn {
    background-color: #1e293b;
    border: 1.5px solid #334155;
    border-radius: 24px;
    min-width: 48px;
    min-height: 48px;
    font-size: 18px;
}

QPushButton.icon-btn:hover {
    background-color: #334155;
    border-color: #38bdf8;
}

/* Карточки букв алфавита */
QPushButton.letter-btn {
    font-size: 22px;
    font-weight: 800;
    border-radius: 10px;
    min-width: 60px;
    min-height: 60px;
    border: 2px solid #334155;
}

QPushButton.letter-locked {
    background-color: #131b2e;
    color: #475569;
    border-color: #1e293b;
}

QPushButton.letter-learning {
    background-color: #2e2611;
    color: #fbbf24;
    border-color: #d97706;
}

QPushButton.letter-learning:hover {
    background-color: #451a03;
}

QPushButton.letter-mastered {
    background-color: #064e3b;
    color: #34d399;
    border-color: #059669;
}

QPushButton.letter-mastered:hover {
    background-color: #065f46;
}

/* Поле поиска */
QLineEdit {
    background-color: #0f172a;
    color: #f8fafc;
    border: 1.5px solid #334155;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 14px;
}

QLineEdit:focus {
    border-color: #38bdf8;
}

/* Шкалы прогресса */
QProgressBar {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    height: 16px;
    text-align: center;
    font-size: 11px;
    font-weight: bold;
    color: #ffffff;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284c7, stop:1 #10b981);
    border-radius: 7px;
}

/* Скроллбары */
QScrollBar:vertical {
    border: none;
    background: #0b0f19;
    width: 10px;
    border-radius: 5px;
}

QScrollBar::handle:vertical {
    background: #334155;
    border-radius: 5px;
    min-height: 25px;
}

QScrollBar::handle:vertical:hover {
    background: #475569;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""

DARK_THEME_QSS = DARK_THEME
