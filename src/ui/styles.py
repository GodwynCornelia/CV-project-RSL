"""
Дизайн-система и стили для патентного GUI стенда непрерывного перевода РЖЯ.
"""

DARK_THEME_QSS = """
QMainWindow {
    background-color: #0b0f19;
    color: #f1f5f9;
}

QWidget {
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    color: #e2e8f0;
}

/* Карточки и контейнеры */
QFrame.panel {
    background-color: #131b2e;
    border: 1px solid #1e293b;
    border-radius: 12px;
}

QFrame.card {
    background-color: #182238;
    border: 1px solid #29354f;
    border-radius: 8px;
    padding: 10px;
}

/* Заголовки */
QLabel.heading {
    font-size: 15px;
    font-weight: 700;
    color: #38bdf8;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

QLabel.subheading {
    font-size: 12px;
    color: #94a3b8;
}

/* Кнопки */
QPushButton.primary-btn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0284c7, stop:1 #0369a1);
    color: #ffffff;
    font-size: 13px;
    font-weight: 600;
    border-radius: 8px;
    padding: 10px 18px;
    border: none;
}

QPushButton.primary-btn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #38bdf8, stop:1 #0284c7);
}

QPushButton.primary-btn:pressed {
    background: #075985;
}

QPushButton.secondary-btn {
    background-color: #1e293b;
    color: #cbd5e1;
    font-size: 13px;
    font-weight: 500;
    border-radius: 8px;
    padding: 9px 16px;
    border: 1px solid #334155;
}

QPushButton.secondary-btn:hover {
    background-color: #334155;
    color: #f8fafc;
    border-color: #475569;
}

QPushButton.accent-btn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #059669, stop:1 #047857);
    color: #ffffff;
    font-size: 13px;
    font-weight: 600;
    border-radius: 8px;
    padding: 10px 18px;
    border: none;
}

QPushButton.accent-btn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #10b981, stop:1 #059669);
}

/* Поле вывода перевода */
QTextEdit.translation-box {
    background-color: #0f172a;
    color: #f8fafc;
    font-size: 18px;
    font-weight: 500;
    line-height: 1.4;
    border: 1.5px solid #0284c7;
    border-radius: 10px;
    padding: 14px;
}

/* Скроллбары */
QScrollBar:vertical {
    border: none;
    background: #0f172a;
    width: 8px;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background: #334155;
    border-radius: 4px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background: #475569;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

/* Бейджи вероятностей */
QLabel.badge-high {
    background-color: #064e3b;
    color: #34d399;
    border: 1px solid #059669;
    border-radius: 4px;
    padding: 3px 6px;
    font-size: 11px;
    font-weight: 600;
}

QLabel.badge-mid {
    background-color: #451a03;
    color: #fbbf24;
    border: 1px solid #d97706;
    border-radius: 4px;
    padding: 3px 6px;
    font-size: 11px;
    font-weight: 600;
}

QLabel.badge-low {
    background-color: #1e293b;
    color: #94a3b8;
    border: 1px solid #334155;
    border-radius: 4px;
    padding: 3px 6px;
    font-size: 11px;
}
"""
