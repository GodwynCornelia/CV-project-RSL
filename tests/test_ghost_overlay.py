import numpy as np

from rsl_trainer.core.dactyl_evaluator import DactylEvaluator
from rsl_trainer.cv.ghost_overlay import GhostOverlayRenderer


def test_scale_translation_and_handedness_do_not_produce_false_warnings():
    template = DactylEvaluator()._get_template_hand("А")
    renderer = GhostOverlayRenderer()
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    box = (300, 70, 220, 320)
    expected = renderer.render_ghost_hand(blank.copy(), template, None, box)
    live = template * [-3, 3] + [6000, 1000]
    actual = renderer.render_ghost_hand(blank.copy(), template, live, box)
    assert np.array_equal(expected, actual)
    y, x = np.nonzero(actual.any(axis=2))
    assert len(x) > 100
    assert x.min() >= box[0] and x.max() <= box[0] + box[2]
    assert y.min() >= box[1] and y.max() <= box[1] + box[3]


def test_calibration_does_not_mutate_shared_camera_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    rendered, _ = GhostOverlayRenderer().draw_calibration_box(frame)
    assert not frame.any()
    assert rendered.any()
