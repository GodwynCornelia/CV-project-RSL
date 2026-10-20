import numpy as np
import pytest

from rsl_trainer.core import dactyl_evaluator as module


@pytest.fixture
def evaluator(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    evaluator = module.DactylEvaluator()
    evaluator.set_target_letter("А")
    return evaluator, clock, evaluator._get_template_hand("А")


def test_continuous_hold_is_required(evaluator):
    evaluator, clock, hand = evaluator
    for now in (0, 0.3, 0.6, 0.9, 1.2):
        clock[0] = now
        assert not evaluator.evaluate_hand(hand)[2]
    clock[0] = 1.5
    score, progress, passed = evaluator.evaluate_hand(hand)
    assert score == 1 and progress == 1 and passed


@pytest.mark.parametrize("missing", [None, np.zeros((21, 2)), np.full((21, 2), np.nan)])
def test_hand_disappearance_resets_hold(evaluator, missing):
    evaluator, clock, hand = evaluator
    evaluator.evaluate_hand(hand)
    clock[0] = 0.4
    evaluator.evaluate_hand(missing)
    clock[0] = 0.7
    assert evaluator.evaluate_hand(hand)[1:] == (0, False)


def test_stalled_camera_cannot_complete_hold(evaluator):
    evaluator, clock, hand = evaluator
    evaluator.evaluate_hand(hand)
    clock[0] = 10
    assert evaluator.evaluate_hand(hand)[1:] == (0, False)


def test_unknown_letter_has_no_generic_successful_template(tmp_path):
    evaluator = module.DactylEvaluator(skeletons_dir=tmp_path)
    evaluator.set_target_letter("unknown")
    assert evaluator._get_template_hand("unknown") is None
    assert evaluator.evaluate_hand(np.ones((21, 2))) == (0, 0, False)


def test_visible_hand_selected_when_middle_frame_is_empty():
    raw = np.zeros((16, 118))
    hand = np.arange(42).reshape(21, 2) + 20
    raw[3, 76:] = hand.ravel()
    assert np.array_equal(module.DactylEvaluator._extract_dominant_hand(raw), hand)


def test_scaled_translated_and_other_leading_hand_match(evaluator):
    evaluator, _, hand = evaluator
    transformed = hand * [-2, 2] + [1300, 300]
    assert evaluator.evaluate_hand(transformed)[0] == pytest.approx(1)
