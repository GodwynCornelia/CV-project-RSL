import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont, QFontDatabase


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    # The offscreen Windows platform does not enumerate system fonts reliably.
    fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    for name in ("segoeui.ttf", "seguisb.ttf", "segoeuib.ttf", "seguiemj.ttf"):
        if (fonts / name).exists():
            QFontDatabase.addApplicationFont(str(fonts / name))
    app.setFont(QFont("Segoe UI", 10))
    yield app


@pytest.fixture
def wait_until(qapp):
    def wait(predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            qapp.processEvents()
            time.sleep(0.005)
        qapp.processEvents()
        assert predicate(), "Timed out waiting for background operation"
    return wait


@pytest.fixture(autouse=True)
def isolate_navigation_test_io(request, monkeypatch, tmp_path):
    """Navigation tests must not access a webcam or the developer's profile."""
    if request.module.__name__ != "test_gui_transitions":
        return
    from rsl_trainer.database.db_manager import DatabaseManager
    from rsl_trainer.ui import main_window
    monkeypatch.setattr(main_window, "DatabaseManager", lambda: DatabaseManager(str(tmp_path / "user.db")))
    monkeypatch.setattr(main_window.CameraWorker, "start", lambda self: None)
    # These tests exercise screens, while inference has its own integration tests.
    monkeypatch.setattr(main_window.GestureInferenceService, "load", lambda self: 1)
