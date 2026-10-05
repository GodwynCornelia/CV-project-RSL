import logging
import threading
import time

logger = logging.getLogger(__name__)


class AudioFeedback:
    """
    Асинхронный звуковой движок обратной связи для доступности (Accessibility)
    и геймификации. Все звуки воспроизводятся в фоновом потоке, исключая лаги GUI.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def _play_async(self, fn) -> None:
        if not self.enabled:
            return
        thread = threading.Thread(target=fn, daemon=True)
        thread.start()

    def play_success(self) -> None:
        """Позитивный мажорный сигнал успешной сдачи жеста/буквы."""
        def _run():
            try:
                import winsound
                winsound.Beep(784, 120)   # G5
                time.sleep(0.04)
                winsound.Beep(1046, 200)  # C6
            except Exception as e:
                logger.debug(f"Audio error: {e}")

        self._play_async(_run)

    def play_error(self) -> None:
        """Мягкий сигнал ошибки/таймаута."""
        def _run():
            try:
                import winsound
                winsound.Beep(260, 220)   # C4 low
            except Exception as e:
                logger.debug(f"Audio error: {e}")

        self._play_async(_run)

    def play_countdown(self) -> None:
        """Короткий щелчок обратного отсчета."""
        def _run():
            try:
                import winsound
                winsound.Beep(523, 70)    # C5
            except Exception as e:
                logger.debug(f"Audio error: {e}")

        self._play_async(_run)

    def play_level_complete(self) -> None:
        """Торжественный аккорд прохождения уровня (Duolingo-style)."""
        def _run():
            try:
                import winsound
                winsound.Beep(523, 100)   # C5
                winsound.Beep(659, 100)   # E5
                winsound.Beep(784, 120)   # G5
                winsound.Beep(1046, 300)  # C6
            except Exception as e:
                logger.debug(f"Audio error: {e}")

        self._play_async(_run)
