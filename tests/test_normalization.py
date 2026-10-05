import numpy as np
import pytest
import torch

from rsl_trainer.core.normalization import (
    normalize_body,
    normalize_hand,
    preprocess_raw_16
)


def test_normalize_hand_zeros():
    """Тест защиты от ZeroDivisionError при полном отсутствии руки (все координаты 0.0)."""
    zeros_hand = np.zeros((21, 2), dtype=np.float32)
    normalized, mask = normalize_hand(zeros_hand)

    assert normalized.shape == (21, 2)
    assert not np.isnan(normalized).any()
    assert not np.isinf(normalized).any()
    assert np.allclose(normalized, 0.0)


def test_normalize_body_zeros():
    """Тест защиты от ZeroDivisionError при отсутствии ключевых точек тела."""
    zeros_body = np.zeros((17, 2), dtype=np.float32)
    normalized = normalize_body(zeros_body)

    assert normalized.shape == (17, 2)
    assert not np.isnan(normalized).any()
    assert not np.isinf(normalized).any()
    assert np.allclose(normalized, 0.0)


def test_normalize_hand_valid():
    """Тест корректной нормализации валидной кисти (центрирование по запястью pt 0)."""
    hand = np.array([
        [100.0, 200.0],  # Запястье
        [105.0, 190.0],
        [110.0, 180.0],
        [115.0, 170.0],
        [120.0, 160.0],
        [100.0, 150.0],
        [100.0, 140.0],
        [100.0, 130.0],
        [100.0, 120.0],  # Средний сустав
        [95.0, 150.0],
        [95.0, 140.0],
        [95.0, 130.0],
        [95.0, 120.0],
        [90.0, 150.0],
        [90.0, 140.0],
        [90.0, 130.0],
        [90.0, 120.0],
        [85.0, 155.0],
        [85.0, 145.0],
        [85.0, 135.0],
        [85.0, 125.0],
    ], dtype=np.float32)

    normalized, mask = normalize_hand(hand)
    assert normalized.shape == (21, 2)
    # Запястье должно стать центром (0, 0)
    assert np.allclose(normalized[0], [0.0, 0.0])
    # Все значения должны лежать в разумном масштабированном диапазоне
    assert np.all(np.abs(normalized) < 10.0)


def test_preprocess_raw_16_shapes():
    """Тест преобразования скользящего окна (16, 118) в 4 входных тензора модели."""
    raw_16 = np.random.randn(16, 118).astype(np.float32)

    b, l, r, m = preprocess_raw_16(raw_16, device="cpu")

    # b: (1, 16, 34)
    assert b.shape == (1, 16, 34)
    # l: (1, 16, 42)
    assert l.shape == (1, 16, 42)
    # r: (1, 16, 42)
    assert r.shape == (1, 16, 42)
    # m: (1, 16, 2)
    assert m.shape == (1, 16, 2)

    assert isinstance(b, torch.Tensor)
    assert not torch.isnan(b).any()
    assert not torch.isnan(l).any()
    assert not torch.isnan(r).any()
    assert not torch.isnan(m).any()
