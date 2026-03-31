import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attendance_system.attendance import AttendanceRunner
from attendance_system.config import Settings, ensure_directories
from attendance_system.db import AttendanceDB
from attendance_system.recognition import FaceRecognitionEngine, PrototypeGallery
from attendance_system.report import format_session_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run classroom attendance from webcam, group image, or video file."
    )
    parser.add_argument(
        "--mode",
        choices=["live", "image", "video"],
        default="live",
        help="Attendance source mode",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index for live mode")
    parser.add_argument("--image-path", type=Path, default=None, help="Path to class group photo")
    parser.add_argument("--video-path", type=Path, default=None, help="Path to class video file")
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Max session duration in seconds. Omit to run until manually stopped.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable OpenCV windows (recommended for remote/headless runs).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='Inference device (e.g. "cpu" or "cuda:0"). Auto-detect when omitted.',
    )
    parser.add_argument("--recognition-threshold", type=float, default=None, help="Match threshold")
    parser.add_argument(
        "--mark-threshold",
        type=float,
        default=None,
        help="Minimum confidence to mark attendance",
    )
    parser.add_argument(
        "--frame-resize",
        type=float,
        default=None,
        help="Resize factor for processed frames (e.g., 0.75)",
    )
    parser.add_argument(
        "--sample-every-n-frames",
        type=int,
        default=None,
        help="Process one out of N frames",
    )
    parser.add_argument(
        "--min-votes",
        type=int,
        default=None,
        help="Frames required to confirm each student in live/video mode",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "image" and args.image_path is None:
        raise SystemExit("--image-path is required when --mode image")
    if args.mode == "video" and args.video_path is None:
        raise SystemExit("--video-path is required when --mode video")
    if args.image_path and not args.image_path.exists():
        raise SystemExit(f"Image not found: {args.image_path}")
    if args.video_path and not args.video_path.exists():
        raise SystemExit(f"Video not found: {args.video_path}")

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    settings = Settings()
    if args.recognition_threshold is not None:
        settings.recognition_threshold = args.recognition_threshold
    if args.mark_threshold is not None:
        settings.mark_threshold = args.mark_threshold
    if args.frame_resize is not None:
        settings.frame_resize = args.frame_resize
    if args.sample_every_n_frames is not None:
        settings.sample_every_n_frames = max(1, args.sample_every_n_frames)
    if args.min_votes is not None:
        settings.min_votes = max(1, args.min_votes)
    ensure_directories(settings)

    db = AttendanceDB(settings.db_path)
    db.init_schema()

    try:
        gallery = PrototypeGallery.from_db(db)
        if gallery.is_empty():
            raise SystemExit(
                "No enrolled student prototypes found. Run enrollment and embedding build first."
            )

        engine = FaceRecognitionEngine(
            gallery=gallery,
            device=device,
            image_size=settings.image_size,
            min_face_size=settings.min_face_size,
            detection_probability=settings.detection_probability,
        )
        runner = AttendanceRunner(db=db, engine=engine, settings=settings)
        display = not args.no_display

        if args.mode == "live":
            result = runner.run_live(
                camera_index=args.camera_index,
                duration_sec=args.duration,
                display=display,
            )
        elif args.mode == "image":
            result = runner.run_group_image(
                image_path=args.image_path,
                display=display,
            )
        else:
            result = runner.run_video_file(
                video_path=args.video_path,
                duration_sec=args.duration,
                display=display,
            )

        print("\nAttendance complete")
        print(f"Session ID         : {result.session_id}")
        print(f"Mode               : {result.mode}")
        print(f"Elapsed (sec)      : {result.elapsed_sec:.2f}")
        print(f"Present count      : {result.present_count}")
        print(f"Processed frames   : {result.processed_frames}")
        print(f"Face detections    : {result.detections}")
        print(f"CSV report         : {result.csv_path}")
        if result.annotated_image_path is not None:
            print(f"Annotated image    : {result.annotated_image_path}")

        print("\nSession Summary")
        print(format_session_summary(db, result.session_id))
    finally:
        db.close()


if __name__ == "__main__":
    main()
