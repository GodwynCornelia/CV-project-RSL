import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np


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

    def set_target_letter(self, letter: str) -> None:
        """Устанавливает текущую целевую букву для тренировки."""
        self._target_letter = letter
        self._hold_start_time = None
        self._is_holding = False

    def _get_template_hand(self, letter: str) -> Optional[np.ndarray]:
        """Возвращает эталонную конфигурацию кисти (21, 2) для буквы."""
        if letter in self._template_cache:
            return self._template_cache[letter]

        # 1. Поиск в кэше skeletons
        template_file = self.skeletons_dir / f"{letter}.npy"
        if template_file.exists():
            tensor = np.load(template_file)
            hand_pts = self._extract_dominant_hand(tensor)
            self._template_cache[letter] = hand_pts
            return hand_pts

        # 2. Поиск в hybrid_tensors/train
        train_letter_dir = Path("hybrid_tensors/train") / letter
        if train_letter_dir.exists():
            files = list(train_letter_dir.glob("*.npy"))
            if files:
                tensor = np.load(files[0])
                hand_pts = self._extract_dominant_hand(tensor)
                self._template_cache[letter] = hand_pts
                return hand_pts

        # 3. Синтетический эталон
        pts = self._create_fallback_hand_template(letter)
        self._template_cache[letter] = pts
        return pts

    @staticmethod
    def _extract_dominant_hand(tensor_16x118: np.ndarray) -> np.ndarray:
        """Извлекает наиболее активную руку (21, 2) из среднего кадра."""
        mid_frame = tensor_16x118[8]
        left_h = mid_frame[34:76].reshape(21, 2)
        right_h = mid_frame[76:118].reshape(21, 2)

        if np.count_nonzero(right_h) > np.count_nonzero(left_h):
            return right_h
        return left_h

    @staticmethod
    def _create_fallback_hand_template(letter: str) -> np.ndarray:
        """Базовый анатомический каркас кисти для демонстрации."""
        base_hand = np.array([
            [0.0, 0.0],       # 0: wrist
            [-0.2, -0.2], [-0.35, -0.4], [-0.45, -0.55], [-0.5, -0.7],  # Thumb
            [-0.15, -0.5], [-0.2, -0.8], [-0.22, -1.0], [-0.23, -1.2],  # Index
            [0.0, -0.52], [0.0, -0.85], [0.0, -1.1], [0.0, -1.3],       # Middle
            [0.15, -0.48], [0.2, -0.8], [0.22, -1.05], [0.23, -1.25],   # Ring
            [0.3, -0.4], [0.38, -0.65], [0.42, -0.85], [0.45, -1.05]    # Pinky
        ], dtype=np.float32)
        return base_hand

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

    def evaluate_hand(self, live_hand_21x2: np.ndarray) -> Tuple[float, float, bool]:
        """
        Оценивает совпадение текущей руки с целевой буквой.
        Возвращает:
        - match_score: float от 0.0 до 1.0 (процент совпадения)
        - hold_progress: float от 0.0 до 1.0 (прогресс удержания 1.5 сек)
        - is_confirmed: bool (True, если жест удержан требуемое время)
        """
        if self._target_letter is None or np.all(live_hand_21x2 == 0):
            self._hold_start_time = None
            self._is_holding = False
            return 0.0, 0.0, False

        template_hand = self._get_template_hand(self._target_letter)
        if template_hand is None or np.all(template_hand == 0):
            return 0.5, 0.0, False

        norm_live = self.normalize_single_hand(live_hand_21x2)
        norm_templ = self.normalize_single_hand(template_hand)

        # Вычисление среднего евклидова расстояния между соответствующими суставами
        dists = np.linalg.norm(norm_live - norm_templ, axis=1)
        mean_dist = float(np.mean(dists))

        # Преобразование расстояния в оценку соответствия (score в [0, 1])
        # При идеальном совпадении mean_dist -> 0, score -> 1.0.
        # При mean_dist = 0.5, score ~ 0.70.
        score = float(np.clip(1.0 - (mean_dist / 1.7), 0.0, 1.0))

        # Оценка удержания конфигурации
        now = time.time()
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
