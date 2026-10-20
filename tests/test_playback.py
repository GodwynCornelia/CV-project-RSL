import numpy as np
import pytest

from rsl_trainer.core.playback import GesturePlayback, interpolate_landmarks


def test_default_demonstration_is_slow_and_holds_ends_without_reverse_motion():
    playback = GesturePlayback(duration=2)
    playback.advance(0.5)
    assert playback.position == 0
    playback.advance(2)
    assert playback.position == pytest.approx(7.5)
    playback.advance(2)
    assert playback.position == 15
    playback.advance(0.8)
    assert playback.position == 15
    playback.advance(0.3)
    assert playback.position == 0


def test_pause_seek_and_speed_change_preserve_pose():
    playback = GesturePlayback()
    playback.seek(6)
    playback.advance(10)
    assert playback.position == 6
    playback.play()
    playback.set_speed(0.25)
    playback.advance(0)
    assert playback.position == pytest.approx(6)
    playback.advance(1)
    assert playback.position == pytest.approx(7.875)


def test_recording_guide_completes_once_in_recording_time():
    playback = GesturePlayback(speed=1)
    playback.configure(16, 3, loop=False)
    playback.advance(1.5)
    assert playback.position == pytest.approx(7.5)
    playback.advance(1.5)
    assert playback.position == 15 and not playback.running
    playback.advance(10)
    assert playback.position == 15


def test_interpolation_is_smooth_without_inventing_missing_hand_positions():
    points = np.array([[[10, 10], [0, 0]], [[20, 30], [40, 50]]], dtype=np.float32)
    valid = np.array([[True, False], [True, True]])
    middle, visible = interpolate_landmarks(points, valid, 0.25)
    assert np.allclose(middle[0], [12.5, 15])
    assert not visible[1] and np.array_equal(middle[1], [0, 0])
    later, visible = interpolate_landmarks(points, valid, 0.75)
    assert visible[1] and np.array_equal(later[1], [40, 50])


@pytest.mark.parametrize("speed", [0, -1, np.nan])
def test_invalid_speeds_rejected(speed):
    with pytest.raises(ValueError):
        GesturePlayback().set_speed(speed)
