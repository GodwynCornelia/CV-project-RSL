from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

from src.config import ModelConfig
from src.models import KinematicState

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# Стандартные связи скелета тела (COCO 17 keypoints)
BODY_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # Голова
    (5, 6),                                   # Плечи
    (5, 7), (7, 9),                           # Левая рука
    (6, 8), (8, 10),                          # Правая рука
    (5, 11), (6, 12), (11, 12),               # Торс
    (11, 13), (13, 15),                       # Левая нога
    (12, 14), (14, 16)                        # Правая нога
]

# Связи пальцев руки (MediaPipe 21 landmarks)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Большой
    (0, 5), (5, 6), (6, 7), (7, 8),          # Указательный
    (0, 9), (9, 10), (10, 11), (11, 12),     # Средний
    (0, 13), (13, 14), (14, 15), (15, 16),   # Безымянный
    (0, 17), (17, 18), (18, 19), (19, 20),   # Мизинец
    (5, 9), (9, 13), (13, 17)                # Основание ладони
]


class HybridTracker:
    """
    Модуль гибридного трекинга (YOLOv8-Pose + MediaPipe Tasks HandLandmarker).
    Извлекает 118 геометрических признаков из каждого видеокадра
    и выполняет эстетичную отрисовку скелета на кадре.
    """

    def __init__(self, config: ModelConfig = ModelConfig()):
        self.config = config
        self._init_models(config.yolo_pose_path, config.hand_landmarker_path)

    def _init_models(self, yolo_path: Union[str, Path], landmarker_path: Union[str, Path]) -> None:
        yolo_path = Path(yolo_path)
        landmarker_path = Path(landmarker_path)

        if not yolo_path.exists():
            raise FileNotFoundError(f"Модель YOLO не найдена: {yolo_path.resolve()}")
        if not landmarker_path.exists():
            raise FileNotFoundError(f"Модель MediaPipe HandLandmarker не найдена: {landmarker_path.resolve()}")

        # 1. Загрузка YOLOv8-pose
        self.yolo_model = YOLO(str(yolo_path))

        # 2. Инициализация MediaPipe HandLandmarker Tasks
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(landmarker_path)),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=2
        )
        self.hand_landmarker = HandLandmarker.create_from_options(options)

    def extract_features(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Извлекает 118 геометрических координат из BGR кадра:
        - 34 координаты скелета тела (YOLO)
        - 42 координаты левой руки (MediaPipe)
        - 42 координаты правой руки (MediaPipe)
        Возвращает np.ndarray формы (118,).
        """
        h, w, _ = frame_bgr.shape

        # 1. Скелет тела через YOLOv8-pose
        yolo_results = self.yolo_model(frame_bgr, verbose=False)
        body_kp = np.zeros(34, dtype=np.float32)

        if yolo_results and len(yolo_results) > 0:
            res = yolo_results[0]
            if res.keypoints is not None and len(res.keypoints.xy) > 0:
                pts = res.keypoints.xy[0].cpu().numpy()
                if pts.shape == (17, 2):
                    body_kp = pts.flatten().astype(np.float32)

        # 2. Кисти рук через MediaPipe Tasks
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        mp_res = self.hand_landmarker.detect(mp_image)

        left_hand = np.zeros(42, dtype=np.float32)
        right_hand = np.zeros(42, dtype=np.float32)

        if mp_res.hand_landmarks and mp_res.handedness:
            for idx, hand_info in enumerate(mp_res.handedness):
                label = hand_info[0].category_name
                landmarks = mp_res.hand_landmarks[idx]

                coords = []
                for lm in landmarks:
                    coords.extend([lm.x * w, lm.y * h])

                arr = np.array(coords, dtype=np.float32)
                if label == 'Left':
                    left_hand = arr
                else:
                    right_hand = arr

        # Конкатенация: 34 + 42 + 42 = 118
        features = np.concatenate([body_kp, left_hand, right_hand]).astype(np.float32)
        return features

    def draw_overlay(
        self,
        frame_bgr: np.ndarray,
        features_118: np.ndarray,
        kinematics: Optional[KinematicState] = None
    ) -> np.ndarray:
        """
        Отрисовывает на кадре скелет тела (неоновый голубой) и пальцы (неоновый зеленый),
        а также HUD-панель кинематических показателей.
        """
        overlay = frame_bgr.copy()

        # Распаковка точек
        body_pts = features_118[:34].reshape(17, 2)
        left_hand_pts = features_118[34:76].reshape(21, 2)
        right_hand_pts = features_118[76:118].reshape(21, 2)

        color_body = (248, 189, 56)      # Neon Cyan / Cyber Blue
        color_left_hand = (128, 222, 74)  # Emerald Green
        color_right_hand = (74, 222, 128) # Lime Green

        # 1. Отрисовка скелета тела
        for pt1_idx, pt2_idx in BODY_CONNECTIONS:
            p1, p2 = body_pts[pt1_idx], body_pts[pt2_idx]
            if (p1[0] > 0 and p1[1] > 0) and (p2[0] > 0 and p2[1] > 0):
                cv2.line(overlay, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color_body, 2, cv2.LINE_AA)

        for pt in body_pts:
            if pt[0] > 0 and pt[1] > 0:
                cv2.circle(overlay, (int(pt[0]), int(pt[1])), 4, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(overlay, (int(pt[0]), int(pt[1])), 5, color_body, 1, cv2.LINE_AA)

        # 2. Отрисовка кистей рук
        for pts, color in [(left_hand_pts, color_left_hand), (right_hand_pts, color_right_hand)]:
            if np.any(pts > 0):
                for pt1_idx, pt2_idx in HAND_CONNECTIONS:
                    p1, p2 = pts[pt1_idx], pts[pt2_idx]
                    if (p1[0] > 0 and p1[1] > 0) and (p2[0] > 0 and p2[1] > 0):
                        cv2.line(overlay, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 2, cv2.LINE_AA)

                for pt in pts:
                    if pt[0] > 0 and pt[1] > 0:
                        cv2.circle(overlay, (int(pt[0]), int(pt[1])), 3, (255, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(overlay, (int(pt[0]), int(pt[1])), 4, color, 1, cv2.LINE_AA)

        # 3. Информационный HUD в углу кадра
        if kinematics is not None:
            phase_text = f"PHASE: {kinematics.phase.value}"
            speed_text = f"VEL: {kinematics.v_ema:.1f} px/f"

            # Цвет индикатора фазы
            phase_color = (180, 180, 180)
            if kinematics.phase.value == "STROKE":
                phase_color = (0, 255, 128)
            elif kinematics.phase.value == "PREPARATION":
                phase_color = (0, 215, 255)
            elif kinematics.phase.value == "TRIGGER":
                phase_color = (0, 69, 255)

            # Полупрозрачная подложка
            cv2.rectangle(overlay, (15, 15), (220, 75), (20, 24, 33), -1)
            cv2.rectangle(overlay, (15, 15), (220, 75), (55, 65, 81), 1)

            cv2.putText(overlay, phase_text, (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, phase_color, 2, cv2.LINE_AA)
            cv2.putText(overlay, speed_text, (25, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        return overlay
