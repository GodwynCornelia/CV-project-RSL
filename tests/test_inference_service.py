import threading

import numpy as np
from PyQt6.QtCore import QTimer

from rsl_trainer.config import AppConfig
from rsl_trainer.core import inference_service as module


def test_load_and_inference_run_in_background_and_share_live_config(qapp, wait_until, monkeypatch):
    gate = threading.Event()
    entered = threading.Event()
    model_threads = []
    instances = []

    class Verifier:
        def __init__(self, config):
            instances.append(self)
            self.config = config
            model_threads.append(threading.get_ident())
            entered.set()
            assert gate.wait(5)

        def verify_target(self, raw, target):
            model_threads.append(threading.get_ident())
            return (target, self.config.dynamic_confidence_threshold)

    monkeypatch.setattr(module, "TargetGestureVerifier", Verifier)
    config = AppConfig()
    service = module.GestureInferenceService(config)
    loaded, results = [], []
    service.loaded.connect(loaded.append)
    service.verified.connect(lambda *args: results.append(args))
    try:
        request = service.load()
        wait_until(entered.is_set)
        # UI timers must still execute while model construction is blocked.
        ticks = []
        QTimer.singleShot(0, lambda: ticks.append(True))
        wait_until(lambda: bool(ticks))
        assert not loaded
        gate.set()
        wait_until(lambda: loaded == [request])
        config.dynamic_confidence_threshold = 0.55
        verify_id = service.verify(np.ones((16, 118)), "вода")
        wait_until(lambda: bool(results))
        assert results == [(verify_id, ("вода", 0.55))]
        assert len(instances) == 1
        assert all(ident != threading.get_ident() for ident in model_threads)
    finally:
        gate.set()
        assert service.shutdown()


def test_load_error_is_reported_and_can_be_retried(qapp, wait_until, monkeypatch):
    attempts = []

    def factory(config):
        attempts.append(True)
        if len(attempts) == 1:
            raise FileNotFoundError("missing checkpoint")
        return object()

    monkeypatch.setattr(module, "TargetGestureVerifier", factory)
    service = module.GestureInferenceService(AppConfig())
    errors, loaded = [], []
    service.failed.connect(lambda *args: errors.append(args))
    service.loaded.connect(loaded.append)
    try:
        request = service.load()
        wait_until(lambda: bool(errors))
        assert errors == [(request, "missing checkpoint")]
        retry = service.load()
        wait_until(lambda: loaded == [retry])
    finally:
        assert service.shutdown()
