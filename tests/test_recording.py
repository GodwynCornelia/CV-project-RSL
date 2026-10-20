import numpy as np
import pytest

from rsl_trainer.core.recording import GestureRecording


def test_sampling_covers_entire_gesture_at_variable_frame_rate():
    recording = GestureRecording(2)
    recording.start(now=10)
    times = [10, 10.1, 10.12, 10.15, 10.4, 10.7, 10.8, 11.2, 11.6, 11.9, 12]
    for now in times:
        recording.append(np.full(118, now), now=now)
    raw = recording.sample()
    assert raw.shape == (16, 118)
    assert raw[0, 0] == 10 and raw[-1, 0] == 12
    assert np.all(np.diff(raw[:, 0]) >= 0)
    assert (raw[:, 0] > 11).sum() >= 6
    assert recording.progress(now=11) == 0.5


@pytest.mark.parametrize("case", ["empty", "no_hands", "interrupted"])
def test_incomplete_or_handless_recording_cannot_be_verified(case):
    recording = GestureRecording(2)
    recording.start(now=0)
    if case != "empty":
        frame = np.ones(118)
        if case == "no_hands":
            frame[34:] = 0
        recording.append(frame, now=0)
        recording.append(frame, now=0.1 if case == "interrupted" else 2)
    with pytest.raises(ValueError):
        recording.sample()


def test_new_attempt_discards_old_frames():
    recording = GestureRecording()
    recording.start(now=0)
    recording.append(np.ones(118), now=1)
    recording.start(now=10)
    assert recording.frames == [] and recording.progress(now=10) == 0
