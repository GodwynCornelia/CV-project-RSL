from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

from rsl_trainer.config import AppConfig
from rsl_trainer.core.model import HierarchicalGestureModel
from rsl_trainer.core.normalization import preprocess_raw_16


@dataclass
class VerificationResult:
    """Результат проверки целевого жеста."""
    is_passed: bool
    target_name: str
    target_confidence: float
    target_rank: int               # Позиция целевого класса в Top-K (1-indexed), или -1 если не вошел
    top_predictions: List[Tuple[str, float]]
    message: str


class TargetGestureVerifier:
    """
    Верификатор целевого жеста (Target-Constrained Verifier).
    Реализует патентный алгоритм бинарной валидации обучающего жеста:
    Passed, если Target in Top-3 и P(Target) >= threshold.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()
        self.device = torch.device(self.config.device)
        self.label2id, self.id2label = self._load_labels(self.config.labels_path)
        self.num_classes = len(self.label2id)

        self.model = HierarchicalGestureModel(num_classes=self.num_classes)
        self._load_checkpoint(self.config.model_path)
        self.model.to(self.device)
        self.model.eval()

    def _load_labels(self, path: Union[str, Path]) -> Tuple[Dict[str, int], Dict[int, str]]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Файл словаря не найден: {path}")
        with open(path, "r", encoding="utf-8") as f:
            label2id = json.load(f)
        id2label = {idx: name for name, idx in label2id.items()}
        return label2id, id2label

    def _load_checkpoint(self, path: Union[str, Path]) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Файл чекпоинта не найден: {path}")
        state = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)

    @torch.inference_mode()
    def predict_top_k(self, raw_16x118: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Возвращает Top-K наиболее вероятных классов и их softmax-вероятности."""
        b, l, r, m = preprocess_raw_16(raw_16x118, device=str(self.device))
        logits = self.model(b, l, r, m)
        probs = F.softmax(logits, dim=-1)[0]

        scores, indices = torch.topk(probs, k=min(top_k, self.num_classes))
        predictions = []
        for score, idx in zip(scores.cpu().numpy(), indices.cpu().numpy()):
            name = self.id2label.get(int(idx), f"cls_{idx}")
            predictions.append((name, float(score)))

        return predictions

    @torch.inference_mode()
    def verify_target(
        self,
        raw_16x118: np.ndarray,
        target_name: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> VerificationResult:
        """
        Проверка выполнения конкретного целевого жеста.
        Target in Top-K и P(Target) >= threshold.
        """
        k = self.config.dynamic_top_k if top_k is None else top_k
        thresh = self.config.dynamic_confidence_threshold if threshold is None else threshold
        raw_16x118 = np.asarray(raw_16x118)
        if raw_16x118.shape != (16, 118):
            raise ValueError(f"Ожидалась форма (16, 118), получена: {raw_16x118.shape}")
        if k < 1 or not 0 <= thresh <= 1:
            raise ValueError("Некорректный порог или количество классов")
        if not any(name.casefold().strip() == target_name.casefold().strip() for name in self.label2id):
            raise ValueError(f"Жест «{target_name}» отсутствует в словаре модели")
        if not np.isfinite(raw_16x118).all() or not np.any(raw_16x118[:, 34:] != 0):
            raise ValueError("Кисти не обнаружены. Покажите жест перед камерой.")

        preds = self.predict_top_k(raw_16x118, top_k=max(k, 10))

        # Поиск целевого класса в предсказаниях
        target_rank = -1
        target_conf = 0.0

        for rank, (name, conf) in enumerate(preds, start=1):
            if name.strip().lower() == target_name.strip().lower():
                target_rank = rank
                target_conf = conf
                break

        # Проверка условия спецификации 3.3
        is_passed = (1 <= target_rank <= k) and (target_conf >= thresh)

        if is_passed:
            msg = f"Отлично! Жест «{target_name}» распознан (Ранг #{target_rank}, {target_conf*100:.1f}%)"
        elif target_rank != -1:
            msg = f"Почти получилось! Уверенность: {target_conf*100:.1f}% (требуется >= {thresh*100:.0f}%)"
        else:
            top1_name, top1_conf = preds[0] if preds else ("?", 0.0)
            msg = f"Попробуйте точнее. Похоже на «{top1_name}» ({top1_conf*100:.1f}%)"

        return VerificationResult(
            is_passed=is_passed,
            target_name=target_name,
            target_confidence=target_conf,
            target_rank=target_rank,
            top_predictions=preds[:k],
            message=msg
        )
