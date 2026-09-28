import json
import logging
import time
from typing import List, Optional
import requests

from src.classifier.lattice import LatticeGraph
from src.config import LLMConfig
from src.models import CandidateLattice, TranslationOutput

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """Ты — специализированный модуль декодирования русского жестового языка (РЖЯ).
На вход подается упорядоченная последовательность распознанных сегментов. Для каждого сегмента дан список Top-5 гипотез с оценками вероятности.
Задача:
1. Выбрать наиболее вероятную цепочку слов, учитывая контекст всего высказывания.
2. Восстановить правильные грамматические окончания, время, предлоги и порядок слов для литературного русского языка.
3. Выдать финальное предложение без лишних комментариев."""


class SemanticDecoder:
    """
    Семантический декодер на базе LLM Qwen.
    Преобразует решетку гипотез изолированных жестовых глосс (Top-5 Lattice)
    в грамматически связное предложение на естественном русском языке.
    """

    def __init__(self, config: LLMConfig = LLMConfig()):
        self.config = config

    def decode_lattice(self, lattices: List[CandidateLattice]) -> TranslationOutput:
        """
        Основной метод декодирования:
        1. Извлекает Top-1 глоссы и строит LatticeGraph с Beam Search.
        2. Формирует JSON-представление решетки.
        3. Запрашивает языковую модель Qwen.
        4. В случае недоступности LLM применяет лингвистический фаллбэк.
        """
        start_time = time.perf_counter()

        if not lattices:
            return TranslationOutput(
                segments_count=0,
                top1_glosses=[],
                lattice=[],
                natural_sentence="",
                latency_ms=0.0
            )

        top1_glosses = [
            lat.top_hypothesis.gloss for lat in lattices
            if lat.top_hypothesis and lat.top_hypothesis.gloss != "no_event"
        ]

        # Оптимизация графа гипотез через Beam Search
        graph = LatticeGraph(lattices)
        beam_paths = graph.beam_search(beam_width=3)
        best_beam_glosses = beam_paths[0].glosses if beam_paths else top1_glosses

        payload_json = graph.to_compact_representation()

        # Попытка генерации через Qwen (Ollama)
        natural_sentence = self._call_qwen_api(payload_json, best_beam_glosses)

        # Если LLM вернула пустоту или сбой, используем интеллектуальный фаллбэк
        if not natural_sentence:
            natural_sentence = self._linguistic_fallback(best_beam_glosses)

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return TranslationOutput(
            segments_count=len(lattices),
            top1_glosses=top1_glosses,
            lattice=lattices,
            natural_sentence=natural_sentence,
            latency_ms=latency_ms
        )

    def _call_qwen_api(self, lattice_json: list, best_beam_glosses: List[str]) -> Optional[str]:
        """Отправка запроса к локальной модели Qwen через Ollama HTTP API."""
        user_message = (
            f"Последовательность сегментов (Top-5 решетка гипотез):\n"
            f"{json.dumps(lattice_json, ensure_ascii=False, indent=2)}\n\n"
            f"Наиболее вероятная цепочка по критерию максимального правдоподобия: {' '.join(best_beam_glosses)}\n\n"
            f"Сформируй связное предложение на литературном русском языке:"
        )

        models_to_try = [self.config.model_name, self.config.fallback_model]

        for model in models_to_try:
            try:
                url = f"{self.config.ollama_host}/api/chat"
                data = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_message}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": 45,  # Ограничение для мгновенной генерации короткой фразы
                    }
                }
                response = requests.post(url, json=data, timeout=self.config.timeout_seconds)
                if response.status_code == 200:
                    result = response.json()
                    raw_text = result.get("message", {}).get("content", "").strip()
                    cleaned = self._clean_llm_output(raw_text)
                    if cleaned:
                        return cleaned
            except Exception as e:
                logger.warning(f"Ошибка запроса к модели {model} через Ollama: {e}")
                continue

        return None

    @staticmethod
    def _clean_llm_output(text: str) -> str:
        """Очистка вывода LLM от лишних служебных символов и кавычек."""
        cleaned = text.strip()
        cleaned = cleaned.strip('"\'`')
        # Удаляем возможное вступительное слово "Предложение: ..."
        for prefix in ["Предложение:", "Перевод:", "Итоговый текст:"]:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        # Капитализация первой буквы и завершающая точка
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]
            if not cleaned.endswith((".", "!", "?")):
                cleaned += "."
        return cleaned

    @staticmethod
    def _linguistic_fallback(glosses: List[str]) -> str:
        """
        Детерминированный лингвистический синтезатор для автономной работы
        без задержек при недоступности внешнего LLM-бэкенда.
        """
        valid_words = [g for g in glosses if g != "no_event" and g.strip()]
        if not valid_words:
            return ""

        text = " ".join(valid_words).capitalize()
        if not text.endswith((".", "!", "?")):
            text += "."
        return text
