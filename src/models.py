from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional
import numpy as np


class KinematicPhase(str, Enum):
    IDLE = "IDLE"                   # Состояние покоя / паузы
    PREPARATION = "PREPARATION"     # Фаза разгона / подъема рук к рабочей зоне
    STROKE = "STROKE"               # Фаза кульминации / активного исполнения жеста
    RETRACTION = "RETRACTION"       # Фаза отвода / замедления движения
    TRIGGER = "TRIGGER"             # Зафиксирована граница жеста


@dataclass
class KinematicState:
    """Метрики кинематики кистей рук на текущем кадре."""
    frame_idx: int
    timestamp: float
    v_raw: float                    # Мгновенная евклидова скорость (пикс/кадр)
    v_ema: float                    # Сглаженная скорость (EMA, alpha=0.3)
    a_t: float                      # Дифференциальное ускорение (v_t - v_{t-1})
    phase: KinematicPhase           # Текущая распознанная фаза
    is_boundary: bool = False       # Сигнал триггера границы завершения жеста
    left_wrist: Optional[Tuple[float, float]] = None
    right_wrist: Optional[Tuple[float, float]] = None


@dataclass
class GestureSegment:
    """Извлеченный сегмент изолированного жеста, приведенный к (16, 118)."""
    segment_id: int
    start_time: float
    end_time: float
    frame_count: int
    tensor: np.ndarray              # Матрица признаков (16, 118)


@dataclass
class Hypothesis:
    """Отдельная гипотеза классификатора для сегмента."""
    gloss: str
    score: float                    # Softmax вероятность [0.0, 1.0]

    def to_dict(self) -> dict:
        return {"gloss": self.gloss, "score": round(self.score, 4)}


@dataclass
class CandidateLattice:
    """Упорядоченная решетка гипотез для одного жеста (Top-K)."""
    segment_id: int
    timestamp: Tuple[float, float]
    candidates: List[Hypothesis] = field(default_factory=list)

    @property
    def top_hypothesis(self) -> Optional[Hypothesis]:
        return self.candidates[0] if self.candidates else None

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "timestamp": [round(self.timestamp[0], 2), round(self.timestamp[1], 2)],
            "candidates": [c.to_dict() for c in self.candidates]
        }


@dataclass
class TranslationOutput:
    """Результат сквозного машинного перевода жестовой цепочки."""
    segments_count: int
    top1_glosses: List[str]
    lattice: List[CandidateLattice]
    natural_sentence: str
    latency_ms: float
