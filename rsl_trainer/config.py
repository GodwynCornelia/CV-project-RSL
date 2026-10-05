from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass
class AppConfig:
    """Глобальная конфигурация приложения «Тренажер РЖЯ»."""
    base_dir: Path = Path(__file__).resolve().parent.parent

    # Пути к обученным весам и словарям
    model_path: Path = base_dir / "best_hierarchical_model.pth"
    labels_path: Path = base_dir / "label2id.json"
    yolo_pose_path: Path = base_dir / "yolov8n-pose.pt"
    hand_landmarker_path: Path = base_dir / "hand_landmarker.task"
    assets_dir: Path = base_dir / "rsl_trainer" / "assets"
    skeletons_dir: Path = base_dir / "rsl_trainer" / "assets" / "skeletons"

    # Параметры оптимизации производительности для CPU
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    inference_width: int = 320        # Разрешение инференса для YOLO на CPU (ускорение в 3-4x)
    inference_height: int = 240
    camera_id: int = 0
    camera_fps: int = 30
    drop_frame_queue_size: int = 1   # Размер очереди: 1 кадр (устраняет накопительный лаг)

    # Параметры валидации дактиля (статичные буквы)
    dactyl_hold_seconds: float = 1.5
    dactyl_threshold: float = 0.70    # Порог уверенности для зачета

    # Параметры валидатора динамических жестов (Target-Constrained Verifier)
    dynamic_window_size: int = 16     # Спецификация 16 кадров
    dynamic_top_k: int = 3            # Top-3 проверка
    dynamic_confidence_threshold: float = 0.30  # P(target) >= 0.30

    # Настройки темы и доступности
    ui_theme: str = "dark"            # dark, light, high_contrast
    sound_enabled: bool = True


# Модульные константы для прямого импорта
ASSETS_DIR: Path = Path(__file__).resolve().parent / "assets"
SKELETONS_DIR: Path = ASSETS_DIR / "skeletons"
