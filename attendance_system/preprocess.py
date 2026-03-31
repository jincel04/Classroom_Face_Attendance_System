from __future__ import annotations

from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN

from .utils import tensor_to_bgr_image


@dataclass
class DetectedFace:
    box: tuple[int, int, int, int]
    probability: float
    face_tensor: torch.Tensor


class FacePreprocessor:
    def __init__(
        self,
        device: str,
        image_size: int = 160,
        min_face_size: int = 40,
        detection_probability: float = 0.90,
        keep_all: bool = True,
    ):
        self.device = device
        self.image_size = image_size
        self.detection_probability = detection_probability
        self.keep_all = keep_all
        self.mtcnn = MTCNN(
            image_size=image_size,
            margin=20,
            min_face_size=min_face_size,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,
            keep_all=keep_all,
            select_largest=not keep_all,
            device=device,
        )

    def detect_and_align(self, frame_bgr: np.ndarray) -> List[DetectedFace]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        boxes, probs = self.mtcnn.detect(rgb)
        if boxes is None or probs is None:
            return []
        if isinstance(boxes, np.ndarray) and boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)
        if np.isscalar(probs):
            probs = np.asarray([float(probs)], dtype=np.float32)
        elif isinstance(probs, np.ndarray) and probs.ndim == 0:
            probs = np.expand_dims(probs.astype(np.float32), axis=0)

        faces = self.mtcnn.extract(rgb, boxes, save_path=None)
        if faces is None:
            return []
        if faces.ndim == 3:
            faces = faces.unsqueeze(0)

        results: List[DetectedFace] = []
        height, width = frame_bgr.shape[:2]
        for box, prob, face_tensor in zip(boxes, probs, faces):
            if prob is None or float(prob) < self.detection_probability:
                continue
            x1, y1, x2, y2 = box
            x1 = max(0, min(width - 1, int(x1)))
            y1 = max(0, min(height - 1, int(y1)))
            x2 = max(0, min(width - 1, int(x2)))
            y2 = max(0, min(height - 1, int(y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            results.append(
                DetectedFace(
                    box=(x1, y1, x2, y2),
                    probability=float(prob),
                    face_tensor=face_tensor,
                )
            )
        return results


def save_aligned_face(face_tensor: torch.Tensor, output_path: str) -> None:
    image = tensor_to_bgr_image(face_tensor)
    cv2.imwrite(output_path, image)
