from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Settings:
    data_dir: Path = PROJECT_ROOT / "data"
    raw_dir: Path = data_dir / "raw"
    aligned_dir: Path = data_dir / "aligned"
    reports_dir: Path = data_dir / "reports"
    sessions_dir: Path = data_dir / "sessions"
    db_path: Path = data_dir / "attendance.db"

    embedding_dim: int = 512
    image_size: int = int(os.getenv("IMAGE_SIZE", "160"))
    min_face_size: int = int(os.getenv("MIN_FACE_SIZE", "40"))
    detection_probability: float = float(os.getenv("DETECTION_PROBABILITY", "0.90"))

    recognition_threshold: float = float(os.getenv("RECOGNITION_THRESHOLD", "0.62"))
    mark_threshold: float = float(os.getenv("MARK_THRESHOLD", "0.68"))
    min_votes: int = int(os.getenv("MIN_VOTES", "2"))

    # Set to 0 (default) to keep attendance running until manually stopped.
    session_duration_sec: int = int(os.getenv("SESSION_DURATION_SEC", "0"))
    frame_resize: float = float(os.getenv("FRAME_RESIZE", "0.75"))
    sample_every_n_frames: int = int(os.getenv("SAMPLE_EVERY_N_FRAMES", "2"))
    # Set to 0 (default) to disable auto-stop when no new students are marked.
    idle_timeout_sec: float = float(os.getenv("IDLE_TIMEOUT_SEC", "0"))


def ensure_directories(settings: Settings) -> None:
    for path in (
        settings.data_dir,
        settings.raw_dir,
        settings.aligned_dir,
        settings.reports_dir,
        settings.sessions_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
