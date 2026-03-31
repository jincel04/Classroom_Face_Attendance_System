from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from facenet_pytorch import fixed_image_standardization

from .config import Settings
from .db import AttendanceDB
from .recognition import EmbeddingModel
from .utils import l2_normalize


VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class EmbeddingBuilder:
    def __init__(
        self,
        db: AttendanceDB,
        settings: Settings,
        device: str,
        batch_size: int = 32,
    ):
        self.db = db
        self.settings = settings
        self.device = device
        self.batch_size = batch_size
        self.embedder = EmbeddingModel(device=device)

    def _image_to_tensor(self, image_path: Path) -> Optional[torch.Tensor]:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            return None
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = cv2.resize(image_rgb, (self.settings.image_size, self.settings.image_size))
        tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float()
        tensor = fixed_image_standardization(tensor)
        return tensor

    def build_for_student(self, student_id: str) -> int:
        student_dir = self.settings.aligned_dir / student_id
        if not student_dir.exists():
            return 0

        image_paths = [p for p in sorted(student_dir.iterdir()) if p.suffix.lower() in VALID_IMAGE_EXTS]
        if not image_paths:
            return 0

        tensors: list[torch.Tensor] = []
        valid_paths: list[Path] = []
        for path in image_paths:
            tensor = self._image_to_tensor(path)
            if tensor is None:
                continue
            tensors.append(tensor)
            valid_paths.append(path)
        if not tensors:
            return 0

        all_embeddings: list[np.ndarray] = []
        for i in range(0, len(tensors), self.batch_size):
            batch_tensors = torch.stack(tensors[i : i + self.batch_size], dim=0)
            batch_embeddings = self.embedder.embed_batch(batch_tensors, use_flip_tta=False)
            all_embeddings.extend(batch_embeddings)

        with self.db.transaction():
            self.db.clear_embeddings(student_id=student_id)
            for emb, path in zip(all_embeddings, valid_paths):
                self.db.add_embedding(
                    student_id=student_id,
                    embedding=emb,
                    source_path=str(path),
                    quality=1.0,
                )

            prototype = l2_normalize(np.mean(np.vstack(all_embeddings), axis=0))
            self.db.set_prototype(student_id=student_id, embedding=prototype)
        return len(all_embeddings)

    def build_all(self, student_id: Optional[str] = None) -> dict[str, int]:
        counts: dict[str, int] = {}
        if student_id:
            counts[student_id] = self.build_for_student(student_id)
            return counts

        students = self.db.list_students()
        for student in students:
            sid = str(student["student_id"])
            counts[sid] = self.build_for_student(sid)
        return counts
