from typing import Any, Dict, Optional
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QStackedWidget, QWidget


class NavigationManager(QStackedWidget):
    """
    Центральный менеджер экранов приложения (MVP Navigator).
    Управляет переходами по карте экранов с сохранением стека истории (Back button).
    """

    screen_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._screens: Dict[str, QWidget] = {}
        self._history = []

    def register_screen(self, name: str, screen_widget: QWidget) -> None:
        """Регистрирует экран в навигаторе."""
        self._screens[name] = screen_widget
        self.addWidget(screen_widget)

    def get_screen(self, name: str) -> Optional[QWidget]:
        return self._screens.get(name)

    def push_screen(self, name: str, **kwargs) -> None:
        """Переходит на экран с добавлением в историю."""
        if name not in self._screens:
            raise KeyError(f"Экран «{name}» не зарегистрирован в навигаторе.")

        target_widget = self._screens[name]
        current_widget = self.currentWidget()

        # Сохраняем текущий экран в стек истории только при смене экрана
        if current_widget is not None and current_widget != target_widget:
            for s_name, s_widget in self._screens.items():
                if s_widget == current_widget:
                    self._history.append(s_name)
                    break

        # Вызов хука подготовки экрана (если определен метод on_enter)
        if hasattr(target_widget, "on_enter"):
            target_widget.on_enter(**kwargs)

        self.setCurrentWidget(target_widget)
        self.screen_changed.emit(name)

    def pop_screen(self) -> None:
        """Возвращается на предыдущий экран из стека истории."""
        if not self._history:
            return

        prev_screen_name = self._history.pop()
        target_widget = self._screens.get(prev_screen_name)
        if target_widget:
            # Вызов хука выхода на текущем экране (если есть on_leave)
            curr = self.currentWidget()
            if hasattr(curr, "on_leave"):
                curr.on_leave()

            if hasattr(target_widget, "on_enter"):
                target_widget.on_enter()

            self.setCurrentWidget(target_widget)
            self.screen_changed.emit(prev_screen_name)

    def navigate_home(self) -> None:
        """Сброс стека и возврат на главный экран."""
        curr = self.currentWidget()
        if hasattr(curr, "on_leave"):
            curr.on_leave()
        self._history.clear()
        self.push_screen("main_menu")
