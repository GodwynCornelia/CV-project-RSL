from dataclasses import dataclass, field
from pathlib import Path
import torch


@dataclass(frozen=True)
class KinematicConfig:
    """Параметры модуля кинематической сегментации жестов (Action Spotting)."""
    ema_alpha: float = 0.3                       # Коэффициент экспоненциального сглаживания скорости
    velocity_threshold_active: float = 16.0     # Порог начала активной фазы жеста (пикселей/кадр)
    velocity_threshold_stop: float = 7.0        # Порог фазы остановки/ретракции
    acceleration_threshold_stop: float = -4.0   # Порог резкого торможения (a_t <= Thresh_stop)
    pause_frames_trigger: int = 5               # Число кадров покоя для фиксации границы
    min_gesture_frames: int = 8                 # Минимальная длительность жеста (отсечение шума)
    max_gesture_frames: int = 80                # Максимальная длительность фазы одного жеста
    sample_frames: int = 16                     # Число кадров в итоговом тензоре жеста (16, 118)
    feature_dim: int = 118                      # 34 (тело) + 42 (левая рука) + 42 (правая рука)


@dataclass(frozen=True)
class ModelConfig:
    """Параметры предобученных моделей и классификатора."""
    base_dir: Path = Path(__file__).resolve().parent.parent
    model_path: Path = field(default_factory=lambda: Path("best_hierarchical_model.pth"))
    labels_path: Path = field(default_factory=lambda: Path("label2id.json"))
    yolo_pose_path: Path = field(default_factory=lambda: Path("yolov8n-pose.pt"))
    hand_landmarker_path: Path = field(default_factory=lambda: Path("hand_landmarker.task"))
    top_k: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class LLMConfig:
    """Параметры семантического языкового декодера Qwen."""
    model_name: str = "qwen2.5:0.5b"            # Легковесная модель для около-реального времени
    fallback_model: str = "qwen2.5:latest"      # Резервная модель, если доступна локально
    ollama_host: str = "http://localhost:11434"
    temperature: float = 0.2
    timeout_seconds: float = 15.0


@dataclass(frozen=True)
class UIConfig:
    """Настройки графического интерфейса пользователя."""
    window_title: str = "RSL Machine Translation — Патентный стенд (MVP)"
    window_width: int = 1440
    window_height: int = 900
    fps: int = 30
    camera_id: int = 0
