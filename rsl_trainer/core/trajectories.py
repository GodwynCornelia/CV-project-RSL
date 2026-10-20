from pathlib import Path

import numpy as np

from rsl_trainer.config import AppConfig


def read_trajectory(path):
    """Read raw pixel coordinates; never substitute an unrelated gesture."""
    try:
        data = np.load(path, allow_pickle=False)
        if (data.ndim == 2 and data.shape[1] == 118 and 2 <= len(data) <= 4096 and np.isfinite(data).all()
                and np.any(data[:, 34:] != 0)):
            return data.astype(np.float32)
    except (OSError, ValueError, TypeError):
        pass
    return None


def find_trajectory(word: str, config=None):
    config = config or AppConfig()
    key = word.strip().casefold()
    roots = [Path(config.skeletons_dir), Path(config.base_dir) / "hybrid_tensors" / "train"]
    for root in roots:
        if not root.is_dir():
            continue
        # Compare names instead of interpreting user text as a filesystem path.
        for entry in root.iterdir():
            if (entry.name if entry.is_dir() else entry.stem).casefold() != key:
                continue
            candidates = [entry] if entry.is_file() else sorted(entry.glob("*.npy"))[:5]
            for candidate in candidates:
                trajectory = read_trajectory(candidate)
                if trajectory is not None:
                    return trajectory
    return None
