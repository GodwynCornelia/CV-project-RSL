import threading

import numpy as np
import pytest

from rsl_trainer.config import AppConfig
from rsl_trainer.cv import camera_worker as module


@pytest.fixture
def camera_fakes(monkeypatch):
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    image[:, :20] = 255
    captures, trackers = [], []

    class Capture:
        def __init__(self, index, backend):
            self.index = index
            self.released = False
            self.opened = True
            self.failing = False
            captures.append(self)

        def set(self, *args):
            pass

        def isOpened(self):
            return self.opened

        def read(self):
            return (False, None) if self.failing else (True, image.copy())

        def release(self):
            self.released = True

    class Tracker:
        def __init__(self, config):
            self.thread_id = threading.get_ident()
            self.inputs = []
            self.closed = False
            trackers.append(self)

        def set_mode(self, mode):
            self.mode = mode

        def extract_features(self, frame):
            self.inputs.append(frame.copy())
            return np.arange(118, dtype=np.float32), np.ones((21, 2)), None

        def draw_skeleton(self, frame, **kwargs):
            return frame.copy()

        def close(self):
            self.closed = True

    monkeypatch.setattr(module.cv2, "VideoCapture", Capture)
    monkeypatch.setattr(module, "FastTracker", Tracker)
    return captures, trackers, image


def test_capture_runs_off_gui_and_mirrors_only_preview(qapp, wait_until, camera_fakes):
    captures, trackers, image = camera_fakes
    worker = module.CameraWorker(AppConfig())
    received = []
    worker.frame_ready.connect(lambda *packet: received.append(packet))
    try:
        worker.start()
        wait_until(lambda: bool(received))
        assert trackers[0].thread_id != threading.get_ident()
        assert np.array_equal(trackers[0].inputs[0], image)
        assert np.array_equal(received[0][0], image[:, ::-1])
        assert np.array_equal(received[0][1], np.arange(118))
        assert worker.stop()
        assert captures[0].released and trackers[0].closed
    finally:
        worker.stop()


def test_latest_frame_replaces_backlog_in_real_worker(qapp):
    worker = module.CameraWorker()
    received = []
    worker.frame_ready.connect(lambda frame, *args: received.append(int(frame[0, 0, 0])))
    for value in range(50):
        worker._publish_frame(np.full((2, 2, 3), value, np.uint8), np.zeros(118), None)
    worker._deliver_latest()
    worker._deliver_latest()
    assert received == [49]


def test_switch_camera_releases_previous_capture(qapp, wait_until, camera_fakes):
    captures, trackers, _ = camera_fakes
    worker = module.CameraWorker()
    try:
        worker.start()
        wait_until(lambda: bool(trackers))
        worker.set_camera_id(2)
        wait_until(lambda: len(trackers) == 2)
        assert captures[0].released and trackers[0].closed
        assert captures[1].index == worker.config.camera_id == 2
        assert worker.isRunning()
    finally:
        worker.stop()


@pytest.mark.parametrize("unplug", [False, True])
def test_camera_failure_reports_error_without_mock_frames(qapp, wait_until, camera_fakes, monkeypatch, unplug):
    captures, trackers, _ = camera_fakes
    if not unplug:
        original = module.cv2.VideoCapture

        def closed_capture(*args):
            cap = original(*args)
            cap.opened = False
            return cap
        monkeypatch.setattr(module.cv2, "VideoCapture", closed_capture)
    worker = module.CameraWorker()
    errors = []
    worker.camera_error.connect(errors.append)
    try:
        worker.start()
        if unplug:
            wait_until(lambda: bool(trackers))
            captures[0].failing = True
        wait_until(lambda: bool(errors))
        wait_until(lambda: not worker.isRunning())
        assert captures[0].released
        if not unplug:
            assert not trackers
    finally:
        worker.stop()
