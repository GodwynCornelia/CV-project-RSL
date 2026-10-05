import sys
import pytest
from PyQt6.QtWidgets import QApplication

from rsl_trainer.config import AppConfig
from rsl_trainer.ui.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app


def test_full_navigation_tree_transitions(qapp):
    """
    Интеграционный тест всего дерева навигации (спецификация 5.3):
    Главный -> Дактиль -> Сетка -> Карточка -> Тренажер -> Возврат назад,
    а также динамические жесты, уровни, инфо и настройки.
    Проверяет корректность стека истории и отсутствие сбоев.
    """
    config = AppConfig()
    window = MainWindow(config=config)
    nav = window.navigator

    # 1. Стартовый экран: main_menu
    assert nav.currentWidget() == nav.get_screen("main_menu")

    # 2. Ветка А: Главный -> Дактиль
    nav.push_screen("dactyl_menu")
    assert nav.currentWidget() == nav.get_screen("dactyl_menu")

    # Дактиль -> Сетка А-Я
    nav.push_screen("alphabet_grid")
    assert nav.currentWidget() == nav.get_screen("alphabet_grid")

    # Сетка -> Карточка буквы 'А'
    nav.push_screen("letter_card_view", letter="А")
    assert nav.currentWidget() == nav.get_screen("letter_card_view")

    # Карточка -> Тренажер буквы 'А'
    nav.push_screen("dactyl_trainer", letter="А")
    assert nav.currentWidget() == nav.get_screen("dactyl_trainer")

    # Обратный откат по стеку
    nav.pop_screen()  # -> letter_card_view
    assert nav.currentWidget() == nav.get_screen("letter_card_view")

    nav.pop_screen()  # -> alphabet_grid
    assert nav.currentWidget() == nav.get_screen("alphabet_grid")

    nav.pop_screen()  # -> dactyl_menu
    assert nav.currentWidget() == nav.get_screen("dactyl_menu")

    nav.pop_screen()  # -> main_menu
    assert nav.currentWidget() == nav.get_screen("main_menu")

    # 3. Экзамен дактиля: Главный -> Дактиль -> Выбор параметров -> Камера-экзамен
    nav.push_screen("dactyl_menu")
    nav.push_screen("test_config")
    assert nav.currentWidget() == nav.get_screen("test_config")

    nav.push_screen("exam_camera", question_count=5)
    assert nav.currentWidget() == nav.get_screen("exam_camera")

    nav.pop_screen()  # -> test_config
    assert nav.currentWidget() == nav.get_screen("test_config")

    nav.navigate_home()
    assert nav.currentWidget() == nav.get_screen("main_menu")

    # 4. Ветка Б: Главный -> Дин. жесты -> Словарь -> Карточка -> Тренажер
    nav.push_screen("dynamic_menu")
    assert nav.currentWidget() == nav.get_screen("dynamic_menu")

    nav.push_screen("word_catalog")
    assert nav.currentWidget() == nav.get_screen("word_catalog")

    nav.push_screen("gesture_card_view", word="Привет!")
    assert nav.currentWidget() == nav.get_screen("gesture_card_view")

    nav.push_screen("gesture_trainer", target_word="Привет!")
    assert nav.currentWidget() == nav.get_screen("gesture_trainer")

    nav.pop_screen()
    assert nav.currentWidget() == nav.get_screen("gesture_card_view")

    nav.navigate_home()
    assert nav.currentWidget() == nav.get_screen("main_menu")

    # 5. Уровни Duolingo: Главный -> Дин. жесты -> Уровни
    nav.push_screen("dynamic_menu")
    nav.push_screen("sentence_levels")
    assert nav.currentWidget() == nav.get_screen("sentence_levels")

    nav.navigate_home()
    assert nav.currentWidget() == nav.get_screen("main_menu")

    # 6. Модальные окна Инфо и Настройки
    nav.push_screen("info_modal")
    assert nav.currentWidget() == nav.get_screen("info_modal")
    nav.pop_screen()
    assert nav.currentWidget() == nav.get_screen("main_menu")

    nav.push_screen("settings_modal")
    assert nav.currentWidget() == nav.get_screen("settings_modal")
    nav.pop_screen()
    assert nav.currentWidget() == nav.get_screen("main_menu")

    # Закрытие окна
    window.close()
