import numpy as np

from rsl_trainer.core.hand_reference import extract_active_hand
from rsl_trainer.core.hand_surface import build_hand_surface, surface_view


def reference_hand():
    return np.array([[0, 0], [-.2, -.1], [-.3, -.2], [-.4, -.3], [-.5, -.4],
                     [-.25, -.7], [-.25, -1], [-.25, -1.3], [-.25, -1.5],
                     [0, -.8], [0, -1.1], [0, -1.4], [0, -1.6],
                     [.2, -.7], [.2, -1], [.2, -1.2], [.2, -1.4],
                     [.4, -.6], [.4, -.85], [.4, -1.1], [.4, -1.2]], np.float32)


def test_signing_hand_selected_over_frequently_visible_idle_hand():
    raw = np.zeros((16, 118), dtype=np.float32)
    idle = reference_hand() * 100 + [400, 800]
    signing = reference_hand() * 100 + [200, 350]
    raw[:, 34:76] = idle.ravel()
    raw[3:13, 76:] = signing.ravel()
    selected = extract_active_hand(raw)
    assert selected.side == 1
    assert np.allclose(selected.points, signing)


def test_stable_pose_selected_instead_of_transient_middle_frame():
    raw = np.zeros((16, 118), dtype=np.float32)
    hand = reference_hand() * 100 + [200, 350]
    raw[:, 76:] = hand.ravel()
    raw[8, 76 + 8 * 2] += 250
    selected = extract_active_hand(raw)
    assert selected.frame != 8
    assert np.allclose(selected.points, hand)


def test_3d_shell_has_volume_and_preserves_observed_2d_joints():
    hand = reference_hand()
    surface = build_hand_surface(hand)
    assert np.allclose(surface.observed_joints[:, :2], hand)
    assert np.all(surface.observed_joints[:, 2] == 0)
    assert not surface.has_source_depth
    assert np.ptp(surface.vertices[:, 2]) > 0
    assert np.isfinite(surface.vertices).all() and np.isfinite(surface.normals).all()
    assert surface.faces.min() >= 0 and surface.faces.max() < len(surface.vertices)
    turned, _ = surface_view(surface, yaw=35, pitch=20)
    assert not np.allclose(turned, surface.vertices)


def test_existing_depth_is_retained_when_available():
    hand = np.column_stack([reference_hand(), np.linspace(0, .3, 21)])
    surface = build_hand_surface(hand)
    assert surface.has_source_depth
    assert np.allclose(surface.observed_joints, hand)
