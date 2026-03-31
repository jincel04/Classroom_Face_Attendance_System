import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attendance_system.config import Settings, ensure_directories
from attendance_system.db import AttendanceDB
from attendance_system.embedding_builder import EmbeddingBuilder
from attendance_system.enrollment import capture_student_from_webcam, enroll_student_from_folder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enroll a student with multiple face images (webcam or folder)."
    )
    parser.add_argument(
        "--student-id",
        required=True,
        help="Unique enrollment number / ID (e.g., 1 or CSE42)",
    )
    parser.add_argument("--name", required=True, help="Student display name")
    parser.add_argument(
        "--source",
        choices=["webcam", "folder"],
        default="webcam",
        help="Enrollment source",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Folder of images (required when --source folder)",
    )
    parser.add_argument("--num-images", type=int, default=25, help="Images to capture via webcam")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index")
    parser.add_argument(
        "--capture-interval",
        type=float,
        default=0.25,
        help="Seconds between captures in webcam mode",
    )
    parser.add_argument(
        "--min-probability",
        type=float,
        default=0.90,
        help="Minimum MTCNN confidence for keeping a face crop",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=640,
        help="Webcam capture width in pixels (webcam mode).",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=480,
        help="Webcam capture height in pixels (webcam mode).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='Inference device (e.g. "cpu" or "cuda:0"). Auto-detect when omitted.',
    )
    parser.add_argument(
        "--skip-embedding-build",
        action="store_true",
        help="Only enroll images; do not rebuild embeddings for this student.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source == "folder" and args.input_dir is None:
        raise SystemExit("--input-dir is required when --source folder")
    if args.source == "folder" and not args.input_dir.exists():
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    settings = Settings()
    ensure_directories(settings)
    db = AttendanceDB(settings.db_path)
    db.init_schema()

    try:
        if args.source == "webcam":
            saved = capture_student_from_webcam(
                db=db,
                settings=settings,
                student_id=args.student_id,
                name=args.name,
                num_images=args.num_images,
                camera_index=args.camera_index,
                capture_interval=args.capture_interval,
                min_probability=args.min_probability,
                camera_width=args.camera_width,
                camera_height=args.camera_height,
                device=device,
            )
        else:
            saved = enroll_student_from_folder(
                db=db,
                settings=settings,
                student_id=args.student_id,
                name=args.name,
                input_dir=args.input_dir,
                min_probability=args.min_probability,
                device=device,
            )

        print(f"Enrollment complete for {args.student_id} ({args.name}). Saved face crops: {saved}")
        if saved == 0:
            print("No valid face crops were saved. Try with better lighting/front-facing images.")
            return

        if not args.skip_embedding_build:
            builder = EmbeddingBuilder(db=db, settings=settings, device=device)
            total = builder.build_for_student(args.student_id)
            print(f"Embeddings built for {args.student_id}: {total}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
