import itertools
from pathlib import Path

import numpy as np
import pytest
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtTest import QTest

from rsl_trainer.config import AppConfig
from rsl_trainer.core.classifier import VerificationResult
from rsl_trainer.database.db_manager import DatabaseManager
from rsl_trainer.ui.screens.dactyl_trainer import DactylTrainerScreen
from rsl_trainer.ui.screens.exam_camera import ExamCameraScreen
from rsl_trainer.ui.screens.gesture_trainer import GestureTrainerScreen
from rsl_trainer.ui.screens.sentence_levels import SentenceLevelsScreen
from rsl_trainer.ui.screens.settings_modal import SettingsScreen
from rsl_trainer.ui.navigation import NavigationManager
from rsl_trainer.ui.theme import DARK_THEME_QSS


class FakeInference(QObject):
    loaded = pyqtSignal(int)
    verified = pyqtSignal(int, object)
    failed = pyqtSignal(int, str)

    def __init__(self):
        super().__init__()
        self.ids = itertools.count(1)
        self.submitted = []

    def load(self):
        return next(self.ids)

    def verify(self, raw, target):
        request = next(self.ids)
        self.submitted.append((request, raw, target))
        return request


class FakeCamera(QObject):
    frame_ready = pyqtSignal(np.ndarray, np.ndarray, object)
    camera_error = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.config = AppConfig(sound_enabled=False)
        self.inference_service = FakeInference()
        self.running = False
        self.camera_id = 0

    def set_mode(self, mode):
        self.mode = mode

    def set_camera_id(self, value):
        self.camera_id = self.config.camera_id = value

    def isRunning(self):
        return self.running

    def start(self):
        self.running = True

    def stop(self):
        self.running = False
        return True


@pytest.fixture
def environment(qapp, tmp_path):
    nav = NavigationManager()
    camera = FakeCamera()
    db = DatabaseManager(str(tmp_path / "progress.db"))
    yield nav, camera, db
    nav.close()


def passed_result(word="вода"):
    return VerificationResult(True, word, 0.31, 2, [(word, 0.31)], "ok")


def fill_recording(screen, raw):
    screen.recording.start(now=0)
    for now, features in zip(np.linspace(0, 2, 16), raw):
        screen.recording.append(features, now=now)


def test_gesture_training_from_countdown_to_saved_mastery(environment, qapp):
    nav, camera, db = environment
    screen = GestureTrainerScreen(nav, camera, db)
    try:
        screen.on_enter(target_word="вода")
        assert not screen.btn_action.isEnabled()
        assert screen.preview_widget.trajectory is not None
        camera.inference_service.loaded.emit(screen._request_id)
        screen._toggle_recording()
        for _ in range(3):
            screen._on_countdown_tick()
        assert screen.state == "RECORDING"
        raw = np.load(camera.config.skeletons_dir / "вода.npy")
        fill_recording(screen, raw)
        screen.recording_timer.stop()
        screen._evaluate_gesture()
        assert screen.state == "EVALUATING" and not screen.btn_action.isEnabled()
        request, submitted, target = camera.inference_service.submitted[-1]
        assert target == "вода" and submitted.shape == (16, 118)
        # Space or another click cannot start a second attempt during evaluation.
        screen._toggle_recording()
        assert screen.state == "EVALUATING"
        camera.inference_service.verified.emit(request, passed_result())
        assert screen.state == "SUCCESS"
        assert db.get_user_stats()["total_xp"] == 30
        assert db.get_gesture_progress("вода")["status"] == "mastered"
        # Duplicate delivery must not award XP twice.
        camera.inference_service.verified.emit(request, passed_result())
        assert db.get_user_stats()["total_xp"] == 30
    finally:
        screen.on_leave()
        screen.close()


def test_late_result_from_previous_visit_cannot_award_xp(environment):
    nav, camera, db = environment
    screen = GestureTrainerScreen(nav, camera, db)
    try:
        screen.on_enter(target_word="вода")
        old_id = screen._request_id
        screen.on_leave()
        screen.on_enter(target_word="дом")
        camera.inference_service.loaded.emit(old_id)
        camera.inference_service.verified.emit(old_id, passed_result())
        assert not screen._model_ready
        assert screen.target_word == "дом"
        assert db.get_user_stats()["total_xp"] == 0
    finally:
        screen.on_leave()


def test_model_load_failure_remains_visible_and_recording_is_disabled(environment):
    nav, camera, db = environment
    screen = GestureTrainerScreen(nav, camera, db)
    try:
        screen.on_enter()
        camera.inference_service.failed.emit(screen._request_id, "missing checkpoint")
        assert "missing checkpoint" in screen.hint_lbl.text()
        assert not screen.btn_action.isEnabled()
        screen._toggle_recording()
        assert screen.state == "IDLE"
    finally:
        screen.on_leave()


@pytest.mark.parametrize("phrase_mode", [False, True])
def test_camera_loss_cancels_recording_until_reconnected(environment, phrase_mode):
    nav, camera, db = environment
    screen_type = SentenceLevelsScreen if phrase_mode else GestureTrainerScreen
    screen = screen_type(nav, camera, db)
    try:
        if phrase_mode:
            screen._start_level(db.get_levels()[0])
        else:
            screen.on_enter()
        camera.inference_service.loaded.emit(screen._request_id)
        screen._start_recording()
        camera.camera_error.emit("disconnected")
        assert screen.state == "IDLE"
        assert not screen.recording_timer.isActive()
        screen._toggle_recording()
        assert screen.state == "IDLE"
        assert not camera.inference_service.submitted
        camera.frame_ready.emit(np.zeros((480, 640, 3), np.uint8), np.zeros(118), None)
        screen._toggle_recording()
        assert screen.state == "COUNTDOWN"
    finally:
        screen.on_leave()


def test_exam_pauses_during_camera_loss_without_counting_failure(environment):
    nav, camera, db = environment
    screen = ExamCameraScreen(nav, camera, db)
    try:
        screen.on_enter(question_count=5)
        camera.camera_error.emit("disconnected")
        assert not screen.timer.isActive()
        assert screen.current_idx == 0 and not screen.failed_letters
        camera.frame_ready.emit(np.zeros((480, 640, 3), np.uint8), np.zeros(118), None)
        assert screen.timer.isActive()
        assert screen.current_idx == 0
    finally:
        screen.on_leave()


def test_sentence_advances_each_verified_step_then_unlocks_next_level(environment):
    nav, camera, db = environment
    screen = SentenceLevelsScreen(nav, camera, db)
    try:
        level = db.get_levels()[0]
        screen._start_level(level)
        camera.inference_service.loaded.emit(screen._request_id)
        for word in level["target_phrase"]:
            assert screen.preview_widget.trajectory is not None
            screen._start_recording()
            raw = np.load(camera.config.skeletons_dir / f"{word}.npy")
            fill_recording(screen, raw)
            screen.recording_timer.stop()
            screen._evaluate_step()
            camera.inference_service.verified.emit(screen._request_id, passed_result(word))
        assert screen.stack.currentWidget() is screen.complete_page
        assert db.get_user_stats()["total_xp"] == 50
        assert db.get_levels()[1]["is_unlocked"] == 1
        assert not camera.isRunning() and not screen.preview_widget.anim_timer.isActive()
    finally:
        screen.on_leave()


@pytest.mark.parametrize("screen_type", [DactylTrainerScreen, ExamCameraScreen])
def test_screens_forward_missing_hand_to_evaluator(environment, monkeypatch, screen_type):
    nav, camera, db = environment
    screen = screen_type(nav, camera, db)
    seen = []
    monkeypatch.setattr(screen.evaluator, "evaluate_hand", lambda hand: seen.append(hand) or (0, 0, False))
    try:
        screen.on_enter()
        camera.frame_ready.emit(np.zeros((480, 640, 3), np.uint8), np.zeros(118), None)
        assert seen and seen[-1] is None
    finally:
        screen.on_leave()


def test_settings_control_camera_duration_and_existing_audio(environment):
    nav, camera, db = environment
    trainer = GestureTrainerScreen(nav, camera, db)
    settings = SettingsScreen(nav, camera.config, camera_worker=camera)
    settings.combo_cam.setCurrentIndex(2)
    settings.slider_thresh.setValue(55)
    settings.spin_duration.setValue(3.5)
    settings.chk_sound.setChecked(True)
    assert camera.camera_id == 2
    assert camera.config.dynamic_confidence_threshold == 0.55
    assert trainer.audio.enabled
    settings.chk_sound.setChecked(False)
    assert not trainer.audio.enabled
    try:
        trainer.on_enter()
        assert trainer.recording.duration == 3.5
    finally:
        trainer.on_leave()


@pytest.mark.parametrize("phrase_mode", [False, True])
def test_silhouette_below_camera_fits_small_window(environment, qapp, phrase_mode):
    nav, camera, db = environment
    screen_type = SentenceLevelsScreen if phrase_mode else GestureTrainerScreen
    screen = screen_type(nav, camera, db)
    host = QMainWindow()
    host.setStyleSheet(DARK_THEME_QSS)
    host.setCentralWidget(screen)
    host.resize(1024, 720)
    try:
        if phrase_mode:
            screen._start_level(db.get_levels()[0])
        else:
            screen.on_enter()
        camera.inference_service.loaded.emit(screen._request_id)
        host.show()
        QTest.qWait(50)
        assert screen.preview_widget.y() >= screen.video_canvas.geometry().bottom()
        assert screen.preview_widget.geometry().bottom() < screen.height()
        button = screen.btn_phrase_action if phrase_mode else screen.btn_action
        assert button.isVisible()
        assert host.width() <= 1024 and host.height() <= 720
        # Keep a reviewable image without adding generated files to the repo.
        output = Path(".pytest_cache") / "previews"
        output.mkdir(parents=True, exist_ok=True)
        filename = "sentence-training.png" if phrase_mode else "gesture-training.png"
        assert host.grab().save(str(output / filename))
    finally:
        screen.on_leave()
        host.close()
