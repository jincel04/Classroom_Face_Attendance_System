from __future__ import annotations

import re
from datetime import datetime, timezone

import cv2
import numpy as np


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def l2_normalize(vec: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < eps:
        return vec.astype(np.float32)
    return (vec / norm).astype(np.float32)


def tensor_to_bgr_image(face_tensor) -> np.ndarray:
    image = face_tensor.detach().cpu().permute(1, 2, 0).numpy()
    # MTCNN post_process outputs approximately [-1, 1].
    if image.min() < 0.0:
        image = (image + 1.0) / 2.0
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_-]+", "_", value)
    return value.strip("_")
