import numpy as np
import pytest

from rsl_trainer.config import AppConfig
from rsl_trainer.cv import fast_tracker as module


def test_yolo_is_loaded_only_once_when_dynamic_tracking_is_requested(tmp_path, monkeypatch):
    weights = tmp_path / "pose.pt"
    weights.touch()
    calls = []

    class Pose:
        def __init__(self, path):
            calls.append(path)

        def __call__(self, frame, **kwargs):
            return []

    monkeypatch.setattr(module, "YOLO", Pose)
    tracker = module.FastTracker.__new__(module.FastTracker)
    tracker.config = AppConfig(yolo_pose_path=weights)
    tracker.yolo_model = tracker.hand_landmarker = None
    frame = np.zeros((480, 640, 3), np.uint8)
    tracker.set_mode(module.TrackerMode.DACTYL)
    features, hand, body = tracker.extract_features(frame)
    assert not calls and features.shape == (118,) and body is None
    tracker.set_mode(module.TrackerMode.DYNAMIC)
    tracker.extract_features(frame)
    tracker.extract_features(frame)
    assert calls == [str(weights)]
    tracker.close()


def test_missing_hand_model_is_reported(tmp_path):
    config = AppConfig(hand_landmarker_path=tmp_path / "missing.task")
    with pytest.raises(FileNotFoundError):
        module.FastTracker(config)


def test_raised_left_hand_is_selected_over_idle_right_hand():
    from types import SimpleNamespace
    hand_left = [SimpleNamespace(x=0.3 + index * 0.001, y=0.3 + index * 0.001) for index in range(21)]
    hand_right = [SimpleNamespace(x=0.7 + index * 0.001, y=0.8 + index * 0.001) for index in range(21)]
    result = SimpleNamespace(hand_landmarks=[hand_left, hand_right],
                             handedness=[[SimpleNamespace(category_name="Left")],
                                         [SimpleNamespace(category_name="Right")]])
    tracker = module.FastTracker.__new__(module.FastTracker)
    tracker.config = AppConfig()
    tracker.hand_landmarker = SimpleNamespace(detect=lambda image: result)
    tracker.yolo_model = None
    tracker.set_mode(module.TrackerMode.DACTYL)
    features, primary, _ = tracker.extract_features(np.zeros((480, 640, 3), np.uint8))
    assert np.allclose(primary, features[34:76].reshape(21, 2))
    assert primary[0, 1] < features[76:118].reshape(21, 2)[0, 1]
