import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
from rsl_trainer.core.hand_reference import extract_active_hand


class DactylEvaluator:
    """
    Вычислитель соответствия конфигурации кисти эталону дактильной буквы (А-Я).
    Сравнивает вектор взаимного расположения пальцев с эталонным шаблоном буквы.
    Реализует условие: непрерывное удержание >= 1.5 сек с уверенностью >= 70%.
    """

    def __init__(self, skeletons_dir: Optional[Path] = None, hold_duration: float = 1.5, threshold: float = 0.70):
        self.hold_duration = hold_duration
        self.threshold = threshold

        if skeletons_dir is None:
            self.skeletons_dir = Path(__file__).resolve().parent.parent / "assets" / "skeletons"
        else:
            self.skeletons_dir = Path(skeletons_dir)

        self._template_cache: Dict[str, np.ndarray] = {}
        self._target_letter: Optional[str] = None
        self._hold_start_time: Optional[float] = None
        self._is_holding: bool = False
        self._last_observation: Optional[float] = None

    def set_target_letter(self, letter: str) -> None:
        """Устанавливает текущую целевую букву для тренировки."""
        self._target_letter = letter
        self._hold_start_time = None
        self._is_holding = False
        self._last_observation = None

    def _get_template_hand(self, letter: str) -> Optional[np.ndarray]:
        """Возвращает эталонную конфигурацию кисти (21, 2) для буквы."""
        if letter in self._template_cache:
            return self._template_cache[letter]

        # Evaluate a few real examples instead of taking an arbitrary middle frame.
        template_file = self.skeletons_dir / f"{letter}.npy"
        candidates = [template_file] if template_file.exists() else []
        train_letter_dir = Path(__file__).resolve().parents[2] / "hybrid_tensors" / "train" / letter
        if train_letter_dir.exists():
            candidates.extend(sorted(train_letter_dir.glob("*.npy"))[:24])
        references = []
        for candidate in candidates:
            try:
                reference = extract_active_hand(np.load(candidate, allow_pickle=False))
                if reference is not None:
                    references.append(reference)
            except (OSError, ValueError):
                continue
        if references:
            points = max(references, key=lambda reference: reference.quality).points
            self._template_cache[letter] = points
            return points
        return None

    @staticmethod
    def _extract_dominant_hand(tensor_16x118: np.ndarray) -> np.ndarray:
        """Извлекает наиболее активную руку (21, 2) из среднего кадра."""
        reference = extract_active_hand(tensor_16x118)
        return reference.points if reference is not None else np.zeros((21, 2), dtype=np.float32)

    @staticmethod
    def normalize_single_hand(hand_21x2: np.ndarray) -> np.ndarray:
        """Нормализует одну кисть: центрирует на запястье и масштабирует по ладони."""
        if np.all(hand_21x2 == 0):
            return hand_21x2.copy()

        wrist = hand_21x2[0]
        centered = hand_21x2 - wrist
        # MCP среднего пальца (точка 9)
        mcp = centered[9] if np.any(centered[9] != 0) else centered[5]
        scale = float(np.linalg.norm(mcp))
        if scale < 1e-4:
            scale = float(np.std(centered)) or 1.0

        return centered / scale

    def evaluate_hand(self, live_hand_21x2: Optional[np.ndarray]) -> Tuple[float, float, bool]:
        """
        Оценивает совпадение текущей руки с целевой буквой.
        Возвращает:
        - match_score: float от 0.0 до 1.0 (процент совпадения)
        - hold_progress: float от 0.0 до 1.0 (прогресс удержания 1.5 сек)
        - is_confirmed: bool (True, если жест удержан требуемое время)
        """
        now = time.monotonic()
        if self._last_observation is not None and now - self._last_observation > 0.5:
            self._hold_start_time = None
        self._last_observation = now
        if (self._target_letter is None or live_hand_21x2 is None
                or np.shape(live_hand_21x2) != (21, 2)
                or not np.isfinite(live_hand_21x2).all() or np.all(live_hand_21x2 == 0)):
            self._hold_start_time = None
            self._is_holding = False
            return 0.0, 0.0, False

        template_hand = self._get_template_hand(self._target_letter)
        if template_hand is None or np.all(template_hand == 0):
            self._hold_start_time = None
            return 0.0, 0.0, False

        norm_live = self.normalize_single_hand(live_hand_21x2)
        norm_templ = self.normalize_single_hand(template_hand)

        # Вычисление среднего евклидова расстояния между соответствующими суставами
        # Accept either leading hand; reflection preserves finger correspondence.
        reflected = norm_live * np.array([-1, 1])
        if np.mean(np.linalg.norm(reflected - norm_templ, axis=1)) < np.mean(np.linalg.norm(norm_live - norm_templ, axis=1)):
            norm_live = reflected
        dists = np.linalg.norm(norm_live - norm_templ, axis=1)
        mean_dist = float(np.mean(dists))

        # Преобразование расстояния в оценку соответствия (score в [0, 1])
        # При идеальном совпадении mean_dist -> 0, score -> 1.0.
        # При mean_dist = 0.5, score ~ 0.70.
        score = float(np.clip(1.0 - (mean_dist / 1.7), 0.0, 1.0))

        # Оценка удержания конфигурации
        is_confirmed = False
        hold_progress = 0.0

        if score >= self.threshold:
            if self._hold_start_time is None:
                self._hold_start_time = now
            elapsed = now - self._hold_start_time
            hold_progress = min(1.0, elapsed / self.hold_duration)
            if hold_progress >= 1.0:
                is_confirmed = True
                self._hold_start_time = None  # Сброс после подтверждения
        else:
            self._hold_start_time = None
            hold_progress = 0.0

        return score, hold_progress, is_confirmed
