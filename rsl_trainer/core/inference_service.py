import itertools
import queue

from PyQt6.QtCore import QThread, pyqtSignal

from rsl_trainer.core.classifier import TargetGestureVerifier


class GestureInferenceService(QThread):
    """One lazy model shared by the trainers, loaded and executed off the GUI.

    Request IDs let screens discard results from a previous visit or attempt.
    """

    loaded = pyqtSignal(int)
    verified = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self._requests = queue.Queue()
        self._ids = itertools.count(1)
        self._verifier = None
        self._closing = False

    def _submit(self, operation, *args):
        if self._closing:
            raise RuntimeError("Распознаватель завершает работу")
        request_id = next(self._ids)
        self._requests.put((request_id, operation, args))
        if not self.isRunning():
            self.start()
        return request_id

    def load(self):
        return self._submit("load")

    def verify(self, raw, target):
        return self._submit("verify", raw.copy(), target)

    def run(self):
        while True:
            request = self._requests.get()
            if request is None:
                return
            request_id, operation, args = request
            try:
                if self._verifier is None:
                    self._verifier = TargetGestureVerifier(self.config)
                if operation == "load":
                    self.loaded.emit(request_id)
                else:
                    self.verified.emit(request_id, self._verifier.verify_target(*args))
            except Exception as exc:
                self.failed.emit(request_id, str(exc))

    def shutdown(self, timeout_ms=5000):
        if not self._closing:
            self._closing = True
            # Discard pending requests; finish the currently executing call.
            while True:
                try:
                    self._requests.get_nowait()
                except queue.Empty:
                    break
            self._requests.put(None)
        return self.wait(timeout_ms)
