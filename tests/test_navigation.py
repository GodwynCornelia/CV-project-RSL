import sys
import pytest
from PyQt6.QtWidgets import QApplication, QLabel, QWidget

from rsl_trainer.ui.navigation import NavigationManager



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


def test_back_restores_selection_and_forward_calls_leave(qapp):
    class Screen(QWidget):
        def __init__(self):
            super().__init__()
            self.params = None
            self.leaves = 0

        def on_enter(self, **kwargs):
            self.params = kwargs

        def on_leave(self):
            self.leaves += 1

    nav = NavigationManager()
    home, card, trainer = Screen(), Screen(), Screen()
    for name, screen in (("main_menu", home), ("card", card), ("trainer", trainer)):
        nav.register_screen(name, screen)
    nav.push_screen("main_menu")
    nav.push_screen("card", word="вода")
    nav.push_screen("trainer", target_word="вода")
    assert card.leaves == 1
    nav.pop_screen()
    assert card.params == {"word": "вода"}
    assert trainer.leaves == 1
    nav.navigate_home()
    assert nav._history == []
    nav.pop_screen()
    assert nav.currentWidget() is home
