import sys
import pytest
from PyQt6.QtWidgets import QApplication, QLabel, QWidget

from rsl_trainer.ui.navigation import NavigationManager


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def test_navigation_flow(qapp):
    nav = NavigationManager()

    screen1 = QLabel("Screen 1")
    screen2 = QLabel("Screen 2")
    screen3 = QLabel("Screen 3")

    nav.register_screen("s1", screen1)
    nav.register_screen("s2", screen2)
    nav.register_screen("s3", screen3)

    # Переход на экран 1
    nav.push_screen("s1")
    assert nav.currentWidget() == screen1

    # Переход на экран 2
    nav.push_screen("s2")
    assert nav.currentWidget() == screen2
    assert nav._history == ["s1"]

    # Переход на экран 3
    nav.push_screen("s3")
    assert nav.currentWidget() == screen3
    assert nav._history == ["s1", "s2"]

    # Pop (возврат на экран 2)
    nav.pop_screen()
    assert nav.currentWidget() == screen2
    assert nav._history == ["s1"]

    # Pop (возврат на экран 1)
    nav.pop_screen()
    assert nav.currentWidget() == screen1
    assert len(nav._history) == 0


def test_on_enter_hooks(qapp):
    nav = NavigationManager()

    class CustomScreen(QWidget):
        def __init__(self):
            super().__init__()
            self.entered_args = None

        def on_enter(self, **kwargs):
            self.entered_args = kwargs

    cust = CustomScreen()
    nav.register_screen("custom", cust)
    nav.push_screen("custom", letter="Ж", mode="test")

    assert cust.entered_args == {"letter": "Ж", "mode": "test"}
