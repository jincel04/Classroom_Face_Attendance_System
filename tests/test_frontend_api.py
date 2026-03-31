import importlib.util
import time
import types
from pathlib import Path


def _load_frontend_server_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_frontend_server.py"
    spec = importlib.util.spec_from_file_location("run_frontend_server_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_rate_limiter_blocks_after_limit():
    server = _load_frontend_server_module()
    limiter = server.RateLimiter(max_requests=2, window_sec=60)
    ok1, _ = limiter.allow("ip-1")
    ok2, _ = limiter.allow("ip-1")
    ok3, retry_after = limiter.allow("ip-1")

    assert ok1 is True
    assert ok2 is True
    assert ok3 is False
    assert retry_after >= 1


def test_live_attendance_manager_start_stop_and_events(monkeypatch):
    server = _load_frontend_server_module()
    manager = server.LiveAttendanceManager()

    def fake_run_loop(self, camera_index: int, device: str):
        with self._lock:
            self._session_id = "session-test"
            self._roster_count = 2
            self._latest_frame_jpeg = b"fake-jpeg"
        self._append_events(
            [
                {
                    "studentId": "S001",
                    "name": "Ada Lovelace",
                    "confidence": 0.93,
                    "detectedAt": "2026-02-18T00:00:00+00:00",
                }
            ]
        )

        while not self._stop_event.is_set():
            with self._lock:
                self._processed_frames += 1
                self._detections += 1
            time.sleep(0.01)

        with self._lock:
            self._csv_path = "fake.csv"
            self._running = False
            self._thread = None
            self._stop_event.clear()

    monkeypatch.setattr(manager, "_run_loop", types.MethodType(fake_run_loop, manager))

    start_status = manager.start(camera_index=3, device="cpu")
    assert start_status["running"] is True
    assert start_status["cameraIndex"] == 3
    assert start_status["device"] == "cpu"

    time.sleep(0.05)
    payload = manager.events(since=0)
    assert payload["ok"] is True
    assert len(payload["events"]) == 1
    assert payload["events"][0]["studentId"] == "S001"
    assert manager.latest_frame() == b"fake-jpeg"

    stop_status = manager.stop(join_timeout_sec=1.0)
    assert stop_status["running"] is False
    assert stop_status["csvPath"] == "fake.csv"


def test_student_storage_helpers_rename_and_delete(tmp_path):
    server = _load_frontend_server_module()
    data_dir = tmp_path / "data"
    settings = server.Settings(
        data_dir=data_dir,
        raw_dir=data_dir / "raw",
        aligned_dir=data_dir / "aligned",
        reports_dir=data_dir / "reports",
        sessions_dir=data_dir / "sessions",
        db_path=data_dir / "attendance.db",
    )
    server.ensure_directories(settings)

    raw_old = settings.raw_dir / "S001"
    aligned_old = settings.aligned_dir / "S001"
    raw_old.mkdir(parents=True, exist_ok=True)
    aligned_old.mkdir(parents=True, exist_ok=True)
    (raw_old / "r.txt").write_text("raw", encoding="utf-8")
    (aligned_old / "a.txt").write_text("aligned", encoding="utf-8")

    server._rename_student_storage_dirs(settings, "S001", "S009")
    assert (settings.raw_dir / "S009" / "r.txt").exists()
    assert (settings.aligned_dir / "S009" / "a.txt").exists()

    removed = server._delete_student_storage_dirs(settings, "S009")
    assert removed == 2
    assert not (settings.raw_dir / "S009").exists()
    assert not (settings.aligned_dir / "S009").exists()
