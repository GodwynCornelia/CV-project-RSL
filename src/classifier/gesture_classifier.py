import json
from pathlib import Path
from typing import Dict, List, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F

from src.classifier.model import HierarchicalGestureModel
from src.config import ModelConfig
from src.models import CandidateLattice, Hypothesis


class GestureClassifier:
    """
    Классификатор изолированных жестов.
    Выполняет иерархическую пространственную нормализацию входного тензора (16, 118)
    и формирует решетку наиболее вероятных гипотез (Top-K Candidate Lattice).
    """

    def __init__(self, config: ModelConfig = ModelConfig()):
        self.config = config
        self.device = torch.device(config.device)
        self.label2id, self.id2label = self._load_labels(config.labels_path)
        self.num_classes = len(self.label2id)

        self.model = HierarchicalGestureModel(num_classes=self.num_classes)
        self._load_checkpoint(config.model_path)
        self.model.to(self.device)
        self.model.eval()

    def _load_labels(self, path: Union[str, Path]) -> Tuple[Dict[str, int], Dict[int, str]]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Файл словаря классов не найден: {path.resolve()}")
        with open(path, "r", encoding="utf-8") as f:
            label2id = json.load(f)
        id2label = {idx: label for label, idx in label2id.items()}
        return label2id, id2label

    def _load_checkpoint(self, path: Union[str, Path]) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Файл чекпоинта модели не найден: {path.resolve()}")
        state_dict = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)

    @staticmethod
    def _normalize_body(body: np.ndarray) -> np.ndarray:
        """
        Нормализует 17 точек скелета относительно плечевого пояса.
        body shape: (16, 17, 2)
        """
        norm_body = np.zeros_like(body)
        for t in range(16):
            pts = body[t]
            # Точки 5 (L Shoulder) и 6 (R Shoulder)
            ls, rs = pts[5], pts[6]
            if np.any(ls != 0) and np.any(rs != 0):
                center = (ls + rs) / 2.0
                scale = np.linalg.norm(ls - rs) + 1e-6
            else:
                non_zero = pts[np.any(pts != 0, axis=1)]
                if len(non_zero) > 0:
                    center = np.mean(non_zero, axis=0)
                    scale = np.std(non_zero) + 1e-6
                else:
                    center = np.zeros(2)
                    scale = 1.0

            valid = np.any(pts != 0, axis=1)
            norm_body[t, valid] = (pts[valid] - center) / scale
        return norm_body

    @staticmethod
    def _normalize_hand(hand: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Центрирует руку по запястью и масштабирует по ладони (wrist -> MCP).
        hand shape: (16, 21, 2)
        returns: (norm_hand, mask (16, 1))
        """
        norm_hand = np.zeros_like(hand)
        mask = np.zeros((16, 1), dtype=np.float32)

        for t in range(16):
            pts = hand[t]
            if np.all(pts == 0):
                continue

            mask[t, 0] = 1.0
            wrist = pts[0]
            mcp = pts[9] if np.any(pts[9] != 0) else pts[5]
            scale = np.linalg.norm(mcp - wrist)
            if scale < 1e-4:
                scale = np.std(pts) + 1e-6

            norm_hand[t] = (pts - wrist) / scale
        return norm_hand, mask

    def preprocess_tensor(self, raw_tensor: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Принимает тензор сырых координат (16, 118) и возвращает нормализованные
        тензоры анатомических подгрупп с добавлением размерности батча (1, ...).
        """
        raw = raw_tensor.astype(np.float32)

        body = raw[:, :34].reshape(16, 17, 2)
        left = raw[:, 34:76].reshape(16, 21, 2)
        right = raw[:, 76:118].reshape(16, 21, 2)

        norm_body = self._normalize_body(body).reshape(1, 16, 34)
        norm_left, l_mask = self._normalize_hand(left)
        norm_right, r_mask = self._normalize_hand(right)

        norm_left = norm_left.reshape(1, 16, 42)
        norm_right = norm_right.reshape(1, 16, 42)
        masks = np.concatenate([l_mask, r_mask], axis=1).reshape(1, 16, 2)

        return (
            torch.tensor(norm_body, dtype=torch.float32, device=self.device),
            torch.tensor(norm_left, dtype=torch.float32, device=self.device),
            torch.tensor(norm_right, dtype=torch.float32, device=self.device),
            torch.tensor(masks, dtype=torch.float32, device=self.device),
        )

    @torch.inference_mode()
    def classify_segment(
        self,
        raw_tensor: np.ndarray,
        segment_id: int = 1,
        timestamp: Tuple[float, float] = (0.0, 0.0),
        top_k: int = 5
    ) -> CandidateLattice:
        """
        Выполняет инференс модели и возвращает CandidateLattice с Top-K гипотезами.
        """
        body, left, right, masks = self.preprocess_tensor(raw_tensor)
        logits = self.model(body, left, right, masks)
        probabilities = F.softmax(logits, dim=-1)[0]

        top_scores, top_indices = torch.topk(probabilities, k=min(top_k, self.num_classes))

        candidates: List[Hypothesis] = []
        for score, idx in zip(top_scores.cpu().numpy(), top_indices.cpu().numpy()):
            gloss = self.id2label.get(int(idx), f"class_{idx}")
            candidates.append(Hypothesis(gloss=gloss, score=float(score)))

        return CandidateLattice(
            segment_id=segment_id,
            timestamp=timestamp,
            candidates=candidates
        )
