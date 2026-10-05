from enum import Enum
from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

from rsl_trainer.config import AppConfig

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


class TrackerMode(str, Enum):
    DACTYL = "DACTYL"       # Оптимизированный режим: только кисти (MediaPipe), YOLO выключен (55+ FPS)
    DYNAMIC = "DYNAMIC"     # Полный режим: скелет (YOLO 320x240) + кисти (MediaPipe) (30+ FPS)


BODY_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # Голова
    (5, 6),                                   # Плечи
    (5, 7), (7, 9),                           # Левая рука
    (6, 8), (8, 10),                          # Правая рука
    (5, 11), (6, 12), (11, 12),               # Торс
    (11, 13), (13, 15),                       # Левая нога
    (12, 14), (14, 16)                        # Правая нога
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Большой
    (0, 5), (5, 6), (6, 7), (7, 8),          # Указательный
    (0, 9), (9, 10), (10, 11), (11, 12),     # Средний
    (0, 13), (13, 14), (14, 15), (15, 16),   # Безымянный
    (0, 17), (17, 18), (18, 19), (19, 20),   # Мизинец
    (5, 9), (9, 13), (13, 17)                # Основание ладони
]


class FastTracker:
    """
    Высокопроизводительный трекер скелетных точек с адаптивным разделением режимов.
    В режиме DACTYL не нагружает CPU моделью YOLO, обеспечивая максимальный FPS.
    В режиме DYNAMIC использует уменьшение масштаба (320x240) для инференса YOLO.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()
        self.mode = TrackerMode.DACTYL

        self.yolo_model: Optional[YOLO] = None
        self.hand_landmarker: Optional[HandLandmarker] = None

        self._init_models()

    def _init_models(self) -> None:
        # 1. Инициализация MediaPipe HandLandmarker Tasks
        mp_path = Path(self.config.hand_landmarker_path)
        if mp_path.exists():
            with open(mp_path, "rb") as f:
                model_bytes = f.read()
            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_buffer=model_bytes),
                running_mode=VisionRunningMode.IMAGE,
                num_hands=2
            )
            self.hand_landmarker = HandLandmarker.create_from_options(options)

        # 2. Инициализация YOLOv8-pose
        yolo_path = Path(self.config.yolo_pose_path)
        if yolo_path.exists():
            self.yolo_model = YOLO(str(yolo_path))

    def set_mode(self, mode: TrackerMode) -> None:
        """Переключает режим трекинга (DACTYL / DYNAMIC)."""
        self.mode = mode

    def extract_features(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Извлекает ключевые точки.
        Возвращает:
        - features_118: (118,) вектор для нейросети
        - primary_hand_21x2: (21, 2) координаты ведущей руки в пикселях кадра
        - body_17x2: (17, 2) координаты скелета тела (или None в режиме DACTYL)
        """
        orig_h, orig_w, _ = frame_bgr.shape
        body_kp = np.zeros((17, 2), dtype=np.float32)

        # 1. YOLOv8-pose (ТОЛЬКО в динамическом режиме для экономии CPU)
        if self.mode == TrackerMode.DYNAMIC and self.yolo_model is not None:
            # Уменьшаем разрешение до 320x240 для 4-кратного ускорения инференса на CPU
            inf_w, inf_h = self.config.inference_width, self.config.inference_height
            small_frame = cv2.resize(frame_bgr, (inf_w, inf_h), interpolation=cv2.INTER_AREA)

            yolo_results = self.yolo_model(small_frame, verbose=False, imgsz=inf_w)
            if yolo_results and len(yolo_results) > 0:
                res = yolo_results[0]
                if res.keypoints is not None and len(res.keypoints.xy) > 0:
                    small_pts = res.keypoints.xy[0].cpu().numpy()
                    if small_pts.shape == (17, 2):
                        # Масштабируем координаты обратно к оригинальному кадру
                        body_kp[:, 0] = small_pts[:, 0] * (orig_w / inf_w)
                        body_kp[:, 1] = small_pts[:, 1] * (orig_h / inf_h)

        # 2. MediaPipe HandLandmarker
        left_hand_pts = np.zeros((21, 2), dtype=np.float32)
        right_hand_pts = np.zeros((21, 2), dtype=np.float32)

        if self.hand_landmarker is not None:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            mp_res = self.hand_landmarker.detect(mp_image)

            if mp_res.hand_landmarks and mp_res.handedness:
                for idx, hand_info in enumerate(mp_res.handedness):
                    label = hand_info[0].category_name
                    landmarks = mp_res.hand_landmarks[idx]

                    coords = np.array([[lm.x * orig_w, lm.y * orig_h] for lm in landmarks], dtype=np.float32)
                    if label == 'Left':
                        left_hand_pts = coords
                    else:
                        right_hand_pts = coords

        # Определение ведущей руки
        if np.count_nonzero(right_hand_pts) > 0:
            primary_hand = right_hand_pts
        elif np.count_nonzero(left_hand_pts) > 0:
            primary_hand = left_hand_pts
        else:
            primary_hand = None

        # Сборка 118 признаков
        features_118 = np.concatenate([
            body_kp.flatten(),
            left_hand_pts.flatten(),
            right_hand_pts.flatten()
        ]).astype(np.float32)

        return features_118, primary_hand, (body_kp if self.mode == TrackerMode.DYNAMIC else None)

    def draw_skeleton(
        self,
        frame_bgr: np.ndarray,
        primary_hand: Optional[np.ndarray] = None,
        body_pts: Optional[np.ndarray] = None,
        color_hand: Tuple[int, int, int] = (74, 222, 128),  # Emerald green
        color_body: Tuple[int, int, int] = (248, 189, 56)   # Neon cyan
    ) -> np.ndarray:
        """Отрисовывает на кадре скелетные линии и ключевые суставы."""
        output = frame_bgr.copy()

        # 1. Отрисовка тела (в динамическом режиме)
        if body_pts is not None and np.any(body_pts > 0):
            for i1, i2 in BODY_CONNECTIONS:
                p1, p2 = body_pts[i1], body_pts[i2]
                if (p1[0] > 0 and p1[1] > 0) and (p2[0] > 0 and p2[1] > 0):
                    cv2.line(output, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color_body, 2, cv2.LINE_AA)
            for p in body_pts:
                if p[0] > 0 and p[1] > 0:
                    cv2.circle(output, (int(p[0]), int(p[1])), 4, color_body, -1, cv2.LINE_AA)

        # 2. Отрисовка руки
        if primary_hand is not None and np.any(primary_hand > 0):
            for i1, i2 in HAND_CONNECTIONS:
                p1, p2 = primary_hand[i1], primary_hand[i2]
                if (p1[0] > 0 and p1[1] > 0) and (p2[0] > 0 and p2[1] > 0):
                    cv2.line(output, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color_hand, 2, cv2.LINE_AA)
            for p in primary_hand:
                if p[0] > 0 and p[1] > 0:
                    cv2.circle(output, (int(p[0]), int(p[1])), 4, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(output, (int(p[0]), int(p[1])), 5, color_hand, 1, cv2.LINE_AA)

        return output
