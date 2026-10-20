from dataclasses import dataclass

import numpy as np


@dataclass
class HandReference:
    points: np.ndarray
    side: int
    frame: int
    quality: float


def extract_active_hand(raw):
    """Choose the signing hand and a stable articulated pose from real samples.

    The hand detected most often may be an idle hand at the signer's side.
    Elevation relative to the shoulders distinguishes it from the signing hand.
    No synthetic letter geometry is generated here.
    """
    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape == (21, 2):
        if not np.isfinite(raw).all() or not np.any(raw):
            return None
        return HandReference(raw.copy(), 0, 0, 1.0)
    if raw.ndim != 2 or raw.shape[1] != 118 or not len(raw) or not np.isfinite(raw).all():
        raise ValueError("Некорректный эталон кисти")
    hands = raw[:, 34:].reshape(-1, 2, 21, 2)
    body = raw[:, :34].reshape(-1, 17, 2)
    shoulders = body[:, [5, 6]]
    shoulder_valid = np.any(shoulders != 0, axis=2).all(axis=1)
    shoulder_scale = np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1)
    fallback_scale = max(float(np.ptp(raw.reshape(-1, 2)[:, 1])), 1.0)
    scale = np.where(shoulder_valid & (shoulder_scale > 1), shoulder_scale, fallback_scale)
    shoulder_y = np.where(shoulder_valid, shoulders[:, :, 1].mean(axis=1), 0)
    candidates = []
    for side in (0, 1):
        hand = hands[:, side]
        palm = np.linalg.norm(hand[:, 9] - hand[:, 0], axis=1)
        visible = np.any(hand != 0, axis=(1, 2)) & (palm > 1e-5)
        indices = np.flatnonzero(visible)
        if not len(indices):
            continue
        elevation = (shoulder_y[indices] - hand[indices, 0, 1]) / scale[indices]
        score = float(np.median(elevation)) - 0.2 * (1 - visible.mean())
        candidates.append((score, side, indices, palm))
    if not candidates:
        return None
    _, side, indices, palm = max(candidates, key=lambda candidate: candidate[0])
    # Avoid preparation and release at the edges of the recording where possible.
    middle = indices[(indices >= len(raw) * 0.25) & (indices <= len(raw) * 0.75)]
    if len(middle):
        indices = middle
    shapes = (hands[indices, side] - hands[indices, side, :1]) / palm[indices, None, None]
    distances = np.linalg.norm(shapes[:, None] - shapes[None, :], axis=-1).mean(axis=-1)
    stability = np.median(distances, axis=1)
    center_bias = abs(indices - (len(raw) - 1) / 2) / len(raw) * 0.03
    chosen = int(np.argmin(stability + center_bias))
    frame = int(indices[chosen])
    coverage = np.any(hands[:, side] != 0, axis=(1, 2)).mean()
    # Prefer a readable projection when several recordings show the same letter.
    # A nearly edge-on palm collapses to a line even with accurate tracking.
    selected = hands[frame, side]
    projected_width = np.linalg.norm(selected[5] - selected[17]) / palm[frame]
    view_penalty = 0.25 * max(0.0, 0.55 - float(projected_width))
    quality = float(coverage - 0.15 * stability[chosen] - center_bias[chosen] - view_penalty)
    return HandReference(hands[frame, side].copy(), side, frame, quality)
