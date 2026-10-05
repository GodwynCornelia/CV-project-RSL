from typing import Tuple
import numpy as np
import torch


def normalize_body(body: np.ndarray) -> np.ndarray:
    """
    Анатомическая нормализация скелета тела (17 точек) относительно плечевого пояса.
    body shape: (T, 17, 2), где T обычно 16.
    Гарантирует отсутствие ZeroDivisionError при пустых/нулевых кадрах.
    """
    if body.ndim == 2:
        body = body[np.newaxis, ...]
        is_single = True
    else:
        is_single = False

    T = body.shape[0]
    norm_body = np.zeros_like(body, dtype=np.float32)

    for t in range(T):
        pts = body[t]
        ls, rs = pts[5], pts[6]  # Левое и правое плечо

        # Проверка наличия плеч
        if np.any(ls != 0) and np.any(rs != 0):
            shoulder_dist = float(np.linalg.norm(ls - rs))
            if shoulder_dist > 1e-4:
                center = (ls + rs) / 2.0
                scale = shoulder_dist
            else:
                center = ls
                scale = 1.0
        else:
            # Фаллбэк по всем ненулевым точкам
            non_zero = pts[np.any(pts != 0, axis=1)]
            if len(non_zero) > 0:
                center = np.mean(non_zero, axis=0)
                std_val = float(np.std(non_zero))
                scale = std_val if std_val > 1e-4 else 1.0
            else:
                center = np.zeros(2, dtype=np.float32)
                scale = 1.0

        valid = np.any(pts != 0, axis=1)
        if scale < 1e-6:
            scale = 1.0

        norm_body[t, valid] = (pts[valid] - center) / scale

    return norm_body[0] if is_single else norm_body


def normalize_hand(hand: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Центрирует кисть по запястью и масштабирует по ладони (wrist -> MCP).
    hand shape: (T, 21, 2).
    Возвращает: (norm_hand, mask shape (T, 1)).
    Полностью защищена от деления на ноль при отсутствии руки.
    """
    if hand.ndim == 2:
        hand = hand[np.newaxis, ...]
        is_single = True
    else:
        is_single = False

    T = hand.shape[0]
    norm_hand = np.zeros_like(hand, dtype=np.float32)
    mask = np.zeros((T, 1), dtype=np.float32)

    for t in range(T):
        pts = hand[t]
        if np.all(pts == 0):
            continue

        mask[t, 0] = 1.0
        wrist = pts[0]  # Точка 0: запястье

        # Основание среднего пальца (MCP, точка 9) или указательного (точка 5)
        mcp = pts[9] if np.any(pts[9] != 0) else pts[5]
        palm_size = float(np.linalg.norm(mcp - wrist))

        if palm_size > 1e-4:
            scale = palm_size
        else:
            std_val = float(np.std(pts))
            scale = std_val if std_val > 1e-4 else 1.0

        if scale < 1e-6:
            scale = 1.0

        norm_hand[t] = (pts - wrist) / scale

    if is_single:
        return norm_hand[0], mask[0]
    return norm_hand, mask


def preprocess_raw_16(raw_16x118: np.ndarray, device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Предобработка сырого 16-кадрового тензора (16, 118) в анатомические тензоры
    с добавлением размерности батча (1, ...).
    """
    raw = raw_16x118.astype(np.float32)
    if raw.shape != (16, 118):
        raise ValueError(f"Ожидалась форма (16, 118), получена: {raw.shape}")

    body = raw[:, :34].reshape(16, 17, 2)
    left = raw[:, 34:76].reshape(16, 21, 2)
    right = raw[:, 76:118].reshape(16, 21, 2)

    norm_b = normalize_body(body).reshape(1, 16, 34)
    norm_l, l_mask = normalize_hand(left)
    norm_r, r_mask = normalize_hand(right)

    norm_l = norm_l.reshape(1, 16, 42)
    norm_r = norm_r.reshape(1, 16, 42)
    masks = np.concatenate([l_mask, r_mask], axis=1).reshape(1, 16, 2)

    dev = torch.device(device)
    return (
        torch.tensor(norm_b, dtype=torch.float32, device=dev),
        torch.tensor(norm_l, dtype=torch.float32, device=dev),
        torch.tensor(norm_r, dtype=torch.float32, device=dev),
        torch.tensor(masks, dtype=torch.float32, device=dev),
    )
