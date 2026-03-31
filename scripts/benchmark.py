import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attendance_system.config import Settings, ensure_directories
from attendance_system.db import AttendanceDB
from attendance_system.recognition import FaceRecognitionEngine, PrototypeGallery


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark recognition throughput on webcam/video.")
    parser.add_argument("--video-path", type=Path, default=None, help="Optional video file for benchmarking")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index if video path is not provided")
    parser.add_argument("--max-frames", type=int, default=120, help="Number of frames to process")
    parser.add_argument("--frame-resize", type=float, default=0.75, help="Resize factor for inference")
    parser.add_argument("--sample-every-n-frames", type=int, default=2, help="Process one of every N frames")
    parser.add_argument(
        "--device",
        default=None,
        help='Inference device (e.g. "cpu" or "cuda:0"). Auto-detect when omitted.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    sample_every_n = max(1, args.sample_every_n_frames)

    settings = Settings()
    ensure_directories(settings)
    db = AttendanceDB(settings.db_path)
    db.init_schema()

    try:
        gallery = PrototypeGallery.from_db(db)
        if gallery.is_empty():
            raise SystemExit("No student prototypes found. Enroll and build embeddings first.")

        engine = FaceRecognitionEngine(
            gallery=gallery,
            device=device,
            image_size=settings.image_size,
            min_face_size=settings.min_face_size,
            detection_probability=settings.detection_probability,
        )

        if args.video_path is not None:
            cap = cv2.VideoCapture(str(args.video_path))
            source = f"video:{args.video_path}"
        else:
            cap = cv2.VideoCapture(args.camera_index)
            source = f"camera:{args.camera_index}"
        if not cap.isOpened():
            raise SystemExit(f"Cannot open benchmark source: {source}")

        latencies_ms = []
        processed = 0
        frame_idx = 0
        total_detections = 0
        started = time.time()

        while processed < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % sample_every_n != 0:
                continue

            if args.frame_resize < 1.0:
                frame = cv2.resize(
                    frame,
                    None,
                    fx=args.frame_resize,
                    fy=args.frame_resize,
                    interpolation=cv2.INTER_LINEAR,
                )

            t0 = time.time()
            matches = engine.recognize_frame(
                frame,
                threshold=settings.recognition_threshold,
                use_flip_tta=False,
            )
            t1 = time.time()
            latencies_ms.append((t1 - t0) * 1000.0)
            total_detections += len(matches)
            processed += 1

        cap.release()
        elapsed = time.time() - started
        if not latencies_ms:
            raise SystemExit("No frames were processed in benchmark.")

        avg_ms = statistics.mean(latencies_ms)
        p95_ms = sorted(latencies_ms)[int(0.95 * (len(latencies_ms) - 1))]
        throughput = processed / elapsed if elapsed > 0 else 0.0

        print("Benchmark complete")
        print(f"Source               : {source}")
        print(f"Device               : {device}")
        print(f"Frames processed     : {processed}")
        print(f"Total elapsed (sec)  : {elapsed:.2f}")
        print(f"Mean latency (ms)    : {avg_ms:.2f}")
        print(f"P95 latency (ms)     : {p95_ms:.2f}")
        print(f"Throughput (fps)     : {throughput:.2f}")
        print(f"Face detections      : {total_detections}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
