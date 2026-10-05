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

    # Сетка -> Карточка буквы 'А' через оба алиаса (letter_card и letter_card_view)
    nav.push_screen("letter_card", letter="А")
    assert nav.currentWidget() == nav.get_screen("letter_card")
    nav.pop_screen()

    nav.push_screen("letter_card_view", letter="Б")
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

    # 6. Модальные окна Инфо и Настройки (оба алиаса)
    nav.push_screen("info")
    assert nav.currentWidget() == nav.get_screen("info")
    nav.pop_screen()

    nav.push_screen("info_modal")
    assert nav.currentWidget() == nav.get_screen("info_modal")
    nav.pop_screen()
    assert nav.currentWidget() == nav.get_screen("main_menu")

    nav.push_screen("settings")
    assert nav.currentWidget() == nav.get_screen("settings")
    nav.pop_screen()

    nav.push_screen("settings_modal")
    assert nav.currentWidget() == nav.get_screen("settings_modal")
    nav.pop_screen()
    assert nav.currentWidget() == nav.get_screen("main_menu")

    # Закрытие окна
    window.close()


def test_patch_v1_2_features(qapp):
    """
    Тестирование ключевых изменений Patch v1.2:
    1. Отсутствие NameError (QPen, QBrush, QPainter) при рендере скелета буквы.
    2. Чистый режим Duolingo-дерева в разделе предложений.
    3. Категоризация, Live Search и сетка карточек в каталоге динамических жестов.
    """
    config = AppConfig()
    window = MainWindow(config=config)
    nav = window.navigator

    # 1. Проверка отсутствия NameError в letter_card_view
    letter_scr = nav.get_screen("letter_card")
    letter_scr.on_enter(letter="А")
    letter_scr.preview_widget.repaint()

    # 2. Проверка раздела предложений (SentenceLevelsScreen)
    levels_scr = nav.get_screen("sentence_levels")
    levels_scr.on_enter()
    assert levels_scr.stack.currentWidget() == levels_scr.tree_page
    levels = levels_scr.db.get_levels()
    assert len(levels) >= 3

    # 3. Проверка каталога динамических жестов
    assert nav.get_screen("gesture_catalog") == nav.get_screen("word_catalog")
    catalog = nav.get_screen("word_catalog")
    assert len(catalog.all_words) in (965, 966)
    assert "top_50" in catalog.categories_map
    assert len(catalog.categories_map["top_50"]) == 50

    # Фильтр по категории "Еда и напитки"
    catalog._on_category_selected("food")
    assert catalog.selected_category == "food"
    assert len(catalog.filtered_words) == len(catalog.categories_map["food"])

    # Живой поиск (Live Search)
    catalog._on_search_changed("чай")
    assert catalog.search_query == "чай"
    assert all("чай" in w.lower() for w in catalog.filtered_words)

    # Буквенный рубрикатор
    catalog._on_search_changed("")
    catalog._on_category_selected("all")
    catalog._on_rubric_selected("В")
    assert catalog.selected_letter == "В"
    assert all(w.lower().startswith("в") for w in catalog.filtered_words)

    # Сброс рубрикатора
    catalog._on_rubric_selected(None)
    assert catalog.selected_letter is None

    window.close()

