import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class TextToSpeechEngine:
    """
    Асинхронный движок Text-To-Speech (TTS) для озвучивания перевода.
    Запускается в фоновом потоке, не блокируя GUI.
    """

    def __init__(self):
        self._engine = None
        self._lock = threading.Lock()
        self._init_engine()

    def _init_engine(self) -> None:
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", 160)      # Скорость речи
            self._engine.setProperty("volume", 0.95)   # Громкость
        except Exception as e:
            logger.warning(f"Не удалось инициализировать pyttsx3: {e}")
            self._engine = None

    def speak(self, text: str) -> None:
        """Озвучивает переданный текст в отдельном потоке (fire-and-forget)."""
        if not text or not text.strip():
            return

        thread = threading.Thread(target=self._speak_worker, args=(text,), daemon=True)
        thread.start()

    def _speak_worker(self, text: str) -> None:
        with self._lock:
            try:
                if self._engine is not None:
                    self._engine.say(text)
                    self._engine.runAndWait()
                else:
                    # Windows PowerShell System.Speech fallback
                    import subprocess
                    clean_text = text.replace("'", "").replace('"', '')
                    ps_cmd = (
                        f"Add-Type -AssemblyName System.Speech; "
                        f"$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                        f"$synth.Speak('{clean_text}')"
                    )
                    subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd], check=False)
            except Exception as e:
                logger.error(f"Ошибка воспроизведения речи TTS: {e}")
