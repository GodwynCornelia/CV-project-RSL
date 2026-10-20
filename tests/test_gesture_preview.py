from pathlib import Path

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QPointF, Qt
from PyQt6.QtGui import QMouseEvent

from rsl_trainer.config import AppConfig
from rsl_trainer.core.trajectories import find_trajectory, read_trajectory
from rsl_trainer.ui.components.gesture_preview import GestureSilhouettePreview, fit_points


def test_pixel_coordinates_fit_canvas_and_preserve_motion():
    points = np.array([[[1200, 500], [1300, 600]], [[1500, 500], [1600, 600]]], np.float32)
    screen, visible = fit_points(points, 400, 200, padding=20, mirror=True)
    assert visible.all()
    assert screen[..., 0].min() >= 20 and screen[..., 0].max() <= 380
    assert screen[..., 1].min() >= 20 and screen[..., 1].max() <= 180
    # Mirrored motion must remain visible rather than recentering every frame.
    assert screen[1, 0, 0] < screen[0, 0, 0] - 100
    assert np.isclose(np.linalg.norm(screen[0, 1] - screen[0, 0]), np.linalg.norm(screen[1, 1] - screen[1, 0]))


def test_real_reference_silhouette_is_visible_and_animates(qapp):
    preview = GestureSilhouettePreview()
    preview.resize(700, 220)
    raw = find_trajectory("Привет!")
    assert raw is not None
    try:
        preview.set_trajectory(raw)
        points, valid = preview.screen_points()
        assert np.isfinite(points[valid]).all()
        assert points[valid][:, 0].min() >= 15
        assert points[valid][:, 0].max() <= preview.width() - 15
        assert points[valid][:, 1].min() >= 15
        assert points[valid][:, 1].max() <= preview.height() - 38 - 15
        image = preview.grab().toImage()
        silhouette_pixels = sum(
            image.pixelColor(x, y).green() > 170 and image.pixelColor(x, y).blue() > 150
            for y in range(0, image.height() - 38, 2) for x in range(0, image.width(), 2)
        )
        assert silhouette_pixels > 20
        assert preview.anim_timer.isActive() and preview.anim_timer.interval() <= 34
        preview.next_button.click()
        assert preview.frame_idx == 1
        assert not preview.anim_timer.isActive()
        preview.set_trajectory(None)
        assert preview.trajectory is None and not preview.anim_timer.isActive()
    finally:
        preview.stop_animation()
        preview.close()


def test_all_bundled_trajectories_stay_inside_preview(qapp):
    preview = GestureSilhouettePreview()
    preview.resize(500, 200)
    try:
        for path in AppConfig().skeletons_dir.glob("*.npy"):
            preview.set_trajectory(path)
            assert preview.trajectory is not None, path.name
            points, valid = preview.screen_points()
            assert np.all(points[valid] >= 0), path.name
            assert points[valid][:, 0].max() <= 500, path.name
            assert points[valid][:, 1].max() <= 162, path.name
    finally:
        preview.stop_animation()


def test_lookup_falls_back_to_dataset_and_rejects_invalid_files(tmp_path):
    config = AppConfig(base_dir=tmp_path, skeletons_dir=tmp_path / "skeletons")
    config.skeletons_dir.mkdir()
    np.save(config.skeletons_dir / "вода.npy", np.zeros((16, 118)))
    folder = tmp_path / "hybrid_tensors" / "train" / "вода"
    folder.mkdir(parents=True)
    raw = np.ones((16, 118)) * 100
    np.save(folder / "sample.npy", raw)
    assert np.array_equal(find_trajectory("ВОДА", config), raw)
    assert find_trajectory("../../anything", config) is None
    bad = tmp_path / "bad.npy"
    np.save(bad, np.full((16, 118), np.nan))
    assert read_trajectory(bad) is None
    preview_data = find_trajectory("unknown", config)
    assert preview_data is None


def test_playback_controls_pause_seek_and_change_speed(qapp):
    preview = GestureSilhouettePreview()
    try:
        preview.set_trajectory(find_trajectory("Привет!"))
        preview.play_button.click()
        assert not preview.playback.running
        preview.phase_slider.setValue(7)
        assert preview.frame_idx == 7
        preview.next_button.click()
        assert preview.frame_idx == 8
        preview.previous_button.click()
        assert preview.frame_idx == 7
        preview.speed_combo.setCurrentIndex(0)
        assert preview.playback.speed == 0.25
        preview.play_button.click()
        assert preview.playback.running and preview.anim_timer.isActive()
        preview.view_combo.setCurrentIndex(1)
        assert not preview.volume_mode
    finally:
        preview.stop_animation()


def test_demonstration_speed_and_duration_restored_after_recording(qapp):
    config = AppConfig(dynamic_recording_seconds=5, demonstration_speed=0.25)
    preview = GestureSilhouettePreview(config=config)
    try:
        preview.set_trajectory(find_trajectory("Привет!"))
        assert preview.playback.movement_duration == 8
        preview.begin_recording(5)
        assert preview.playback.movement_duration == 5
        assert not preview.controls.isEnabled()
        preview.playback.advance(2.5)
        preview.finish_recording()
        assert preview.playback.position == pytest.approx(7.5)
        assert not preview.playback.running and preview.controls.isEnabled()
        assert preview.playback.movement_duration == 8
        assert config.dynamic_recording_seconds == 5
        preview.speed_combo.setCurrentIndex(1)
        other = GestureSilhouettePreview(config=config)
        other.set_trajectory(find_trajectory("Привет!"))
        assert other.playback.speed == 0.5
        other.stop_animation()
    finally:
        preview.stop_animation()


def test_longer_demonstrations_are_not_truncated_to_classifier_window(qapp, tmp_path):
    raw = np.ones((64, 118), dtype=np.float32)
    path = tmp_path / "detailed.npy"
    np.save(path, raw)
    preview = GestureSilhouettePreview()
    try:
        preview.set_trajectory(path)
        assert len(preview.trajectory) == 64
        assert preview.phase_slider.maximum() == 63
        preview.seek_phase(63)
        assert preview.frame_idx == 63
        preview.next_button.click()
        assert preview.frame_idx == 63
    finally:
        preview.stop_animation()


def test_letter_volume_can_be_rotated_and_reset(qapp):
    from rsl_trainer.core.dactyl_evaluator import DactylEvaluator
    from rsl_trainer.ui.screens.letter_card_view import HandSkeletonPreview
    preview = HandSkeletonPreview()
    preview.resize(300, 320)
    preview.set_hand(DactylEvaluator()._get_template_hand("А"))
    original = preview.grab().toImage()
    press = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(100, 100), QPointF(100, 100),
                        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    move = QMouseEvent(QEvent.Type.MouseMove, QPointF(150, 125), QPointF(150, 125),
                       Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
    preview.mousePressEvent(press)
    preview.mouseMoveEvent(move)
    assert preview.yaw != 0 and preview.pitch != 0
    assert preview.grab().toImage() != original
    preview.mouseDoubleClickEvent(press)
    assert preview.yaw == preview.pitch == 0
    output = Path(".pytest_cache/previews")
    output.mkdir(parents=True, exist_ok=True)
    assert preview.grab().save(str(output / "letter-volume.png"))
