import numpy as np

import attendance_system.attendance as attendance_module
from attendance_system.attendance import AttendanceRunner
from attendance_system.config import Settings, ensure_directories
from attendance_system.db import AttendanceDB
from attendance_system.recognition import MatchResult, PrototypeGallery
from attendance_system.utils import l2_normalize


class _FakeWebcamStream:
    def __init__(self, camera_index: int = 0):
        self.camera_index = camera_index

    def start(self):
        return self

    def read(self):
        return np.zeros((120, 160, 3), dtype=np.uint8)

    def stop(self):
        return None


class _FakeEngine:
    def __init__(self):
        self.gallery = PrototypeGallery(
            student_ids=["S001"],
            names=["Ada Lovelace"],
            prototypes=np.zeros((1, 512), dtype=np.float32),
        )

    def recognize_frame(self, frame_bgr, threshold: float, use_flip_tta: bool = False):
        return [
            MatchResult(
                box=(10, 12, 70, 90),
                student_id="S001",
                name="Ada Lovelace",
                confidence=0.95,
            )
        ]


def _test_settings(tmp_path):
    data_dir = tmp_path / "data"
    return Settings(
        data_dir=data_dir,
        raw_dir=data_dir / "raw",
        aligned_dir=data_dir / "aligned",
        reports_dir=data_dir / "reports",
        sessions_dir=data_dir / "sessions",
        db_path=tmp_path / "attendance_test.db",
    )


def test_live_runner_marks_and_stops_on_manual_quit(monkeypatch, tmp_path):
    settings = _test_settings(tmp_path)
    settings.sample_every_n_frames = 1
    settings.frame_resize = 1.0
    settings.min_votes = 1
    settings.recognition_threshold = 0.5
    settings.mark_threshold = 0.5
    settings.idle_timeout_sec = 0
    settings.session_duration_sec = 0
    ensure_directories(settings)

    db = AttendanceDB(settings.db_path)
    db.init_schema()
    db.upsert_student("S001", "Ada Lovelace")
    emb = l2_normalize(np.random.rand(512).astype(np.float32))
    db.set_prototype("S001", emb)

    monkeypatch.setattr(attendance_module, "WebcamStream", _FakeWebcamStream)
    monkeypatch.setattr(attendance_module.cv2, "imshow", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(attendance_module.cv2, "waitKey", lambda *_args, **_kwargs: ord("q"))
    monkeypatch.setattr(attendance_module.cv2, "destroyAllWindows", lambda: None)

    runner = AttendanceRunner(db=db, engine=_FakeEngine(), settings=settings)
    result = runner.run_live(camera_index=0, duration_sec=None, display=True)

    try:
        assert result.present_count == 1
        assert result.processed_frames >= 1
        assert result.csv_path.exists()
        rows = db.get_session_records(result.session_id)
        assert len(rows) == 1
        assert rows[0]["student_id"] == "S001"
    finally:
        db.close()
