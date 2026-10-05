from typing import Optional, Tuple
import cv2
import numpy as np


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # Большой
    (0, 5), (5, 6), (6, 7), (7, 8),          # Указательный
    (0, 9), (9, 10), (10, 11), (11, 12),     # Средний
    (0, 13), (13, 14), (14, 15), (15, 16),   # Безымянный
    (0, 17), (17, 18), (18, 19), (19, 20),   # Мизинец
    (5, 9), (9, 13), (13, 17)                # Основание
]

POSE_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # Голова / уши
    (5, 6),                                  # Плечи
    (5, 7), (7, 9),                          # Левая рука
    (6, 8), (8, 10),                         # Правая рука
    (5, 11), (6, 12), (11, 12),              # Торс
    (11, 13), (13, 15),                      # Левая нога
    (12, 14), (14, 16)                       # Правая нога
]


class GhostOverlayRenderer:
    """
    Рендерер полупрозрачного эталонного каркаса (Ghost Overlay) в соответствии со Спецификацией 3.2.
    - Прозрачность 40-50%
    - Масштабирование под габариты калибровочной рамки
    - Динамическая цветовая индикация отклонения Delta_err:
      зеленый при совпадении, оранжевый/янтарный при отклонении пальцев.
    """

    HAND_CONNECTIONS = HAND_CONNECTIONS
    POSE_CONNECTIONS = POSE_CONNECTIONS

    def __init__(self, delta_err_thresh: float = 35.0):
        self.delta_err_thresh = delta_err_thresh

    def draw_calibration_box(self, frame_bgr: np.ndarray, box: Optional[Tuple[int, int, int, int]] = None) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Отрисовывает габаритную рамку калибровки (угловые метки для руки) в правой верхней зоне.
        box: (x, y, w, h)
        """
        h, w, _ = frame_bgr.shape
        if box is None:
            # Дефолтная зона: справа по центру
            bw = int(w * 0.38)
            bh = int(h * 0.55)
            bx = int(w * 0.56)
            by = int(h * 0.22)
            box = (bx, by, bw, bh)

        bx, by, bw, bh = box
        corner_len = 24
        col = (200, 200, 200)  # Бело-серый
        thick = 2

        # 4 уголка
        # Верхний левый
        cv2.line(frame_bgr, (bx, by), (bx + corner_len, by), col, thick)
        cv2.line(frame_bgr, (bx, by), (bx, by + corner_len), col, thick)
        # Верхний правый
        cv2.line(frame_bgr, (bx + bw, by), (bx + bw - corner_len, by), col, thick)
        cv2.line(frame_bgr, (bx + bw, by), (bx + bw, by + corner_len), col, thick)
        # Нижний левый
        cv2.line(frame_bgr, (bx, by + bh), (bx + corner_len, by + bh), col, thick)
        cv2.line(frame_bgr, (bx, by + bh), (bx, by + bh - corner_len), col, thick)
        # Нижний правый
        cv2.line(frame_bgr, (bx + bw, by + bh), (bx + bw - corner_len, by + bh), col, thick)
        cv2.line(frame_bgr, (bx + bw, by + bh), (bx + bw, by + bh - corner_len), col, thick)

        cv2.putText(frame_bgr, "ЗОНА ДЛЯ РУКИ", (bx + 8, by - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        return frame_bgr, box

    def render_ghost_hand(
        self,
        frame_bgr: np.ndarray,
        template_hand_21x2: np.ndarray,
        live_hand_21x2: Optional[np.ndarray],
        calibration_box: Tuple[int, int, int, int]
    ) -> np.ndarray:
        """
        Отрисовывает эталонный каркас руки с полупрозрачностью и подсветкой отклонений.
        """
        if template_hand_21x2 is None or np.all(template_hand_21x2 == 0):
            return frame_bgr

        bx, by, bw, bh = calibration_box
        target_center_x = bx + bw // 2
        target_center_y = by + int(bh * 0.65)
        scale_px = bh * 0.45

        # Масштабирование шаблона кисти в пиксели кадра
        norm_t = template_hand_21x2.copy()
        wrist = norm_t[0]
        norm_t = norm_t - wrist
        # Нормализуем по MCP
        mcp = norm_t[9] if np.any(norm_t[9] != 0) else norm_t[5]
        s = float(np.linalg.norm(mcp)) or 1.0
        norm_t = (norm_t / s) * scale_px

        ghost_pts = np.zeros((21, 2), dtype=np.int32)
        ghost_pts[:, 0] = (target_center_x + norm_t[:, 0]).astype(np.int32)
        ghost_pts[:, 1] = (target_center_y + norm_t[:, 1]).astype(np.int32)

        # Подложка для полупрозрачности
        overlay = frame_bgr.copy()

        # Оценка локального отклонения каждого сустава
        deviations = [False] * 21
        if live_hand_21x2 is not None and np.any(live_hand_21x2 > 0):
            # Сопоставляем живую руку с призраком
            live_wrist = live_hand_21x2[0]
            live_aligned = live_hand_21x2 - live_wrist + [target_center_x, target_center_y]
            diffs = np.linalg.norm(live_aligned - ghost_pts, axis=1)
            for idx in range(21):
                if diffs[idx] > self.delta_err_thresh:
                    deviations[idx] = True

        # Цвета:
        normal_ghost_color = (255, 200, 80)    # Небесно-голубой эталон
        warning_color = (30, 140, 255)         # Оранжево-янтарный цвет отклонения

        # Отрисовка соединительных линий эталона
        for p1_i, p2_i in self.HAND_CONNECTIONS:
            p1 = tuple(ghost_pts[p1_i])
            p2 = tuple(ghost_pts[p2_i])
            is_deviated = deviations[p1_i] or deviations[p2_i]
            edge_color = warning_color if is_deviated else normal_ghost_color
            cv2.line(overlay, p1, p2, edge_color, 3, cv2.LINE_AA)

        # Отрисовка суставов
        for i, pt in enumerate(ghost_pts):
            p = tuple(pt)
            node_color = warning_color if deviations[i] else normal_ghost_color
            cv2.circle(overlay, p, 4, node_color, -1, cv2.LINE_AA)
            cv2.circle(overlay, p, 6, (255, 255, 255), 1, cv2.LINE_AA)

        # Наложение с альфа-каналом (45% прозрачность эталона)
        alpha = 0.45
        cv2.addWeighted(overlay, alpha, frame_bgr, 1.0 - alpha, 0, frame_bgr)

        return frame_bgr
