from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

from .db import AttendanceDB
from .preprocess import DetectedFace, FacePreprocessor
from .utils import l2_normalize


@dataclass
class MatchResult:
    box: tuple[int, int, int, int]
    student_id: Optional[str]
    name: str
    confidence: float


class EmbeddingModel:
    def __init__(self, device: str):
        self.device = device
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    @torch.inference_mode()
    def embed_batch(self, face_tensors: torch.Tensor, use_flip_tta: bool = False) -> np.ndarray:
        if face_tensors.ndim == 3:
            face_tensors = face_tensors.unsqueeze(0)
        batch = face_tensors.to(self.device)
        emb = self.model(batch)
        if use_flip_tta:
            flipped = torch.flip(batch, dims=[3])
            emb_flip = self.model(flipped)
            emb = (emb + emb_flip) / 2.0
        emb_np = emb.detach().cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(emb_np, axis=1, keepdims=True) + 1e-10
        return emb_np / norms


class PrototypeGallery:
    def __init__(self, student_ids: list[str], names: list[str], prototypes: np.ndarray):
        self.student_ids = student_ids
        self.names = names
        self.prototypes = prototypes.astype(np.float32) if prototypes.size else prototypes

    @classmethod
    def from_db(cls, db: AttendanceDB) -> "PrototypeGallery":
        student_ids, names, matrix = db.load_prototypes()
        if matrix.size:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
            matrix = matrix / norms
        return cls(student_ids=student_ids, names=names, prototypes=matrix)

    def is_empty(self) -> bool:
        return self.prototypes.size == 0 or len(self.student_ids) == 0

    def best_match(self, embedding: np.ndarray, threshold: float) -> tuple[Optional[str], str, float]:
        if self.is_empty():
            return None, "Unknown", 0.0
        query = l2_normalize(embedding)
        scores = self.prototypes @ query
        idx = int(np.argmax(scores))
        score = float(scores[idx])
        if score < threshold:
            return None, "Unknown", score
        return self.student_ids[idx], self.names[idx], score


class FaceRecognitionEngine:
    def __init__(
        self,
        gallery: PrototypeGallery,
        device: str,
        image_size: int = 160,
        min_face_size: int = 40,
        detection_probability: float = 0.90,
    ):
        self.gallery = gallery
        self.preprocessor = FacePreprocessor(
            device=device,
            image_size=image_size,
            min_face_size=min_face_size,
            detection_probability=detection_probability,
        )
        self.embedder = EmbeddingModel(device=device)

    def reload_gallery(self, db: AttendanceDB) -> None:
        self.gallery = PrototypeGallery.from_db(db)

    def _embed_faces(self, faces: list[DetectedFace], use_flip_tta: bool = False) -> np.ndarray:
        tensors = torch.stack([face.face_tensor for face in faces], dim=0)
        return self.embedder.embed_batch(tensors, use_flip_tta=use_flip_tta)

    def recognize_frame(
        self,
        frame_bgr,
        threshold: float,
        use_flip_tta: bool = False,
    ) -> list[MatchResult]:
        faces = self.preprocessor.detect_and_align(frame_bgr)
        if not faces:
            return []
        embeddings = self._embed_faces(faces, use_flip_tta=use_flip_tta)
        results: list[MatchResult] = []
        for face, emb in zip(faces, embeddings):
            student_id, name, confidence = self.gallery.best_match(emb, threshold=threshold)
            results.append(
                MatchResult(
                    box=face.box,
                    student_id=student_id,
                    name=name,
                    confidence=confidence,
                )
            )
        return results

