import time
from typing import Callable, List, Optional, Tuple
import numpy as np

from src.classifier.gesture_classifier import GestureClassifier
from src.config import KinematicConfig, LLMConfig, ModelConfig
from src.decoding.semantic_decoder import SemanticDecoder
from src.models import (
    CandidateLattice,
    GestureSegment,
    KinematicState,
    TranslationOutput
)
from src.spotting.kinematic_spotter import KinematicSpotter
from src.tracking.hybrid_tracker import HybridTracker


class ContinuousSignTranslationPipeline:
    """
    Сквозной пайплайн непрерывного перевода жестовой речи:
    [Видеопоток] -> [Hybrid Tracker] -> [Kinematic Spotter] -> [Classifier] -> [Lattice] -> [Semantic Decoder]
    """

    def __init__(
        self,
        model_config: ModelConfig = ModelConfig(),
        kinematic_config: KinematicConfig = KinematicConfig(),
        llm_config: LLMConfig = LLMConfig(),
        init_tracker: bool = True
    ):
        self.model_config = model_config
        self.kinematic_config = kinematic_config
        self.llm_config = llm_config

        # 1. Трекер ключевых точек (может быть отключен в офлайн тестах с готовыми тензорами)
        self.tracker: Optional[HybridTracker] = None
        if init_tracker:
            self.tracker = HybridTracker(config=model_config)

        # 2. Кинематический споттер фаз и границ
        self.spotter = KinematicSpotter(config=kinematic_config)

        # 3. Классификатор с иерархической нормализацией
        self.classifier = GestureClassifier(config=model_config)

        # 4. Семантический языковой декодер Qwen
        self.decoder = SemanticDecoder(config=llm_config)

        # Буфер распознанных в текущей фразе сегментов и решеток гипотез
        self.recognized_lattices: List[CandidateLattice] = []
        self.last_translation: Optional[TranslationOutput] = None

        # Коллбэки для синхронизации с GUI
        self.on_kinematics_updated: Optional[Callable[[KinematicState], None]] = None
        self.on_lattice_generated: Optional[Callable[[CandidateLattice], None]] = None
        self.on_translation_ready: Optional[Callable[[TranslationOutput], None]] = None

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp: Optional[float] = None
    ) -> Tuple[np.ndarray, KinematicState, Optional[CandidateLattice]]:
        """
        Обработка живого кадра с веб-камеры.
        Возвращает аннотированный кадр, кинематическое состояние и (если зафиксирован жест) решетку гипотез.
        """
        if self.tracker is None:
            raise RuntimeError("Трекер не инициализирован. Запустите пайплайн с init_tracker=True.")

        ts = timestamp if timestamp is not None else time.time()

        # 1. Извлечение 118 признаков
        features_118 = self.tracker.extract_features(frame_bgr)

        # 2. Анализ кинематики и границ
        kinematics, segment = self.spotter.process_frame(features_118, ts)

        # 3. При фиксации границы жеста — классификация
        new_lattice: Optional[CandidateLattice] = None
        if segment is not None:
            new_lattice = self.classify_segment(segment)

        # 4. Отрисовка скелета и HUD
        overlay_frame = self.tracker.draw_overlay(frame_bgr, features_118, kinematics)

        if self.on_kinematics_updated:
            self.on_kinematics_updated(kinematics)

        return overlay_frame, kinematics, new_lattice

    def process_features_frame(
        self,
        features_118: np.ndarray,
        timestamp: float
    ) -> Tuple[KinematicState, Optional[CandidateLattice]]:
        """
        Обработка уже извлеченного вектора признаков (118,) без видеокадра (для тестов и воспроизведения).
        """
        kinematics, segment = self.spotter.process_frame(features_118, timestamp)
        new_lattice: Optional[CandidateLattice] = None

        if segment is not None:
            new_lattice = self.classify_segment(segment)

        if self.on_kinematics_updated:
            self.on_kinematics_updated(kinematics)

        return kinematics, new_lattice

    def classify_segment(self, segment: GestureSegment) -> CandidateLattice:
        """Классификация 16-кадрового сегмента и сохранение в сессионную решетку."""
        lattice = self.classifier.classify_segment(
            raw_tensor=segment.tensor,
            segment_id=segment.segment_id,
            timestamp=(segment.start_time, segment.end_time),
            top_k=self.model_config.top_k
        )
        self.recognized_lattices.append(lattice)

        if self.on_lattice_generated:
            self.on_lattice_generated(lattice)

        return lattice

    def translate_current_phrase(self) -> TranslationOutput:
        """
        Запуск семантического декодирования всей накопленной фразы через Qwen LLM.
        """
        translation = self.decoder.decode_lattice(self.recognized_lattices)
        self.last_translation = translation

        if self.on_translation_ready:
            self.on_translation_ready(translation)

        return translation

    def clear_phrase(self) -> None:
        """Очистить текущую жестовую цепочку."""
        self.recognized_lattices.clear()
        self.last_translation = None
        self.spotter.reset()
