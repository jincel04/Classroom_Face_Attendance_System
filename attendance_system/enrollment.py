from __future__ import annotations

import shutil
import time
from pathlib import Path

import cv2

from .camera import WebcamStream
from .config import Settings
from .db import AttendanceDB
from .preprocess import FacePreprocessor, save_aligned_face


VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _student_dirs(settings: Settings, student_id: str) -> tuple[Path, Path]:
    raw_dir = settings.raw_dir / student_id
    aligned_dir = settings.aligned_dir / student_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    aligned_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, aligned_dir


def capture_student_from_webcam(
    db: AttendanceDB,
    settings: Settings,
    student_id: str,
    name: str,
    num_images: int = 25,
    camera_index: int = 0,
    capture_interval: float = 0.25,
    min_probability: float = 0.90,
    camera_width: int = 640,
    camera_height: int = 480,
    device: str = "cpu",
) -> int:
    db.upsert_student(student_id=student_id, name=name)
    raw_dir, aligned_dir = _student_dirs(settings, student_id)

    preprocessor = FacePreprocessor(
        device=device,
        image_size=settings.image_size,
        min_face_size=settings.min_face_size,
        detection_probability=min_probability,
        keep_all=False,
    )
    stream = WebcamStream(
        camera_index=camera_index,
        width=camera_width,
        height=camera_height,
        fps=30,
    ).start()
    saved = 0
    last_capture = 0.0

    try:
        while saved < num_images:
            frame = stream.read()
            if frame is None:
                continue

            faces = preprocessor.detect_and_align(frame)
            overlay = frame.copy()
            cv2.putText(
                overlay,
                f"Student: {name} ({student_id}) | Saved: {saved}/{num_images}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            best_face = None
            if faces:
                best_face = faces[0]
                x1, y1, x2, y2 = best_face.box
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    overlay,
                    f"{best_face.probability:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            can_capture = (time.time() - last_capture) >= capture_interval
            if best_face is not None and can_capture:
                saved += 1
                timestamp = int(time.time() * 1000)
                raw_path = raw_dir / f"{saved:03d}_{timestamp}.jpg"
                aligned_path = aligned_dir / f"{saved:03d}_{timestamp}.jpg"
                cv2.imwrite(str(raw_path), frame)
                save_aligned_face(best_face.face_tensor, str(aligned_path))
                last_capture = time.time()

            cv2.imshow("Enrollment - press Q to quit", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        stream.stop()
        cv2.destroyAllWindows()

    return saved


def enroll_student_from_folder(
    db: AttendanceDB,
    settings: Settings,
    student_id: str,
    name: str,
    input_dir: Path,
    min_probability: float = 0.90,
    device: str = "cpu",
) -> int:
    db.upsert_student(student_id=student_id, name=name)
    raw_dir, aligned_dir = _student_dirs(settings, student_id)

    preprocessor = FacePreprocessor(
        device=device,
        image_size=settings.image_size,
        min_face_size=settings.min_face_size,
        detection_probability=min_probability,
    )

    files = [p for p in sorted(input_dir.iterdir()) if p.suffix.lower() in VALID_IMAGE_EXTS]
    saved = 0
    for image_path in files:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        faces = preprocessor.detect_and_align(frame)
        if not faces:
            continue
        best_face = max(faces, key=lambda x: x.probability)
        saved += 1
        raw_path = raw_dir / f"{saved:03d}_{image_path.name}"
        aligned_path = aligned_dir / f"{saved:03d}_{image_path.name}"
        shutil.copy2(str(image_path), str(raw_path))
        save_aligned_face(best_face.face_tensor, str(aligned_path))
    return saved
