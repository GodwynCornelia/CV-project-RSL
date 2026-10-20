from pathlib import Path
import numpy as np
import pytest
import torch

from rsl_trainer.config import AppConfig
from rsl_trainer.core.classifier import TargetGestureVerifier
from rsl_trainer.core.model import HierarchicalGestureModel


def test_model_forward():
    """Тест прямого прохода нейросети HierarchicalGestureModel."""
    config = AppConfig()
    model = HierarchicalGestureModel(num_classes=999)
    model.eval()

    b = torch.randn(1, 16, 34)
    l = torch.randn(1, 16, 42)
    r = torch.randn(1, 16, 42)
    m = torch.randn(1, 16, 2)

    with torch.no_grad():
        out = model(b, l, r, m)

    assert out.shape == (1, 999)
    assert not torch.isnan(out).any()


def test_target_verifier():
    """Тест верификатора целевого жеста с синтетическим инференсом."""
    config = AppConfig()
    if not Path(config.model_path).exists() or not Path(config.labels_path).exists():
        pytest.skip("Файлы весов или словаря не найдены")

    verifier = TargetGestureVerifier(config)
    dummy_input = np.zeros((16, 118), dtype=np.float32)

    with pytest.raises(ValueError, match="Кисти не обнаружены"):
        verifier.verify_target(dummy_input, target_name="Привет!")
    raw = np.load(config.skeletons_dir / "Привет!.npy")
    res = verifier.verify_target(raw, target_name="Привет!")
    assert res.target_name == "Привет!"
    assert isinstance(res.is_passed, bool)
    assert len(res.top_predictions) <= 3


def test_verifier_uses_changed_threshold_and_rejects_invalid_input():
    verifier = TargetGestureVerifier.__new__(TargetGestureVerifier)
    verifier.config = AppConfig()
    verifier.label2id = {"вода": 0}
    verifier.predict_top_k = lambda raw, top_k: [("вода", 0.4)]
    raw = np.ones((16, 118), dtype=np.float32)
    assert verifier.verify_target(raw, "вода").is_passed
    verifier.config.dynamic_confidence_threshold = 0.55
    assert not verifier.verify_target(raw, "вода").is_passed
    assert verifier.verify_target(raw, "вода", threshold=0).is_passed
    with pytest.raises(ValueError):
        verifier.verify_target(raw, "unknown")
    with pytest.raises(ValueError):
        verifier.verify_target(raw[:2], "вода")
    with pytest.raises(ValueError):
        verifier.verify_target(np.full((16, 118), np.nan), "вода")
