from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
from collections import defaultdict, deque
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attendance_system.camera import WebcamStream
from attendance_system.config import Settings, ensure_directories
from attendance_system.db import AttendanceDB
from attendance_system.embedding_builder import EmbeddingBuilder
from attendance_system.enrollment import capture_student_from_webcam
from attendance_system.recognition import FaceRecognitionEngine, MatchResult, PrototypeGallery
from attendance_system.utils import utc_now_iso


FRONTEND_DIR = ROOT / "frontend"
ENROLL_LOCK = threading.Lock()
ADMIN_TOKEN = os.getenv("ATTENDANCE_ADMIN_TOKEN", "").strip()
ENROLL_RATE_LIMIT_MAX = int(os.getenv("ENROLL_RATE_LIMIT_MAX", "4"))
ENROLL_RATE_LIMIT_WINDOW_SEC = float(os.getenv("ENROLL_RATE_LIMIT_WINDOW_SEC", "300"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve frontend pages with enrollment and live attendance APIs."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    return parser.parse_args()


def _student_storage_dirs(settings: Settings, student_id: str) -> list[Path]:
    return [
        settings.raw_dir / student_id,
        settings.aligned_dir / student_id,
    ]


def _rename_student_storage_dirs(settings: Settings, old_student_id: str, new_student_id: str) -> None:
    if old_student_id == new_student_id:
        return
    for base_dir in (settings.raw_dir, settings.aligned_dir):
        src = base_dir / old_student_id
        dst = base_dir / new_student_id
        if not src.exists():
            continue
        if dst.exists():
            for item in src.iterdir():
                target = dst / item.name
                if target.exists():
                    continue
                item.rename(target)
            try:
                src.rmdir()
            except OSError:
                pass
            continue
        src.rename(dst)


def _delete_student_storage_dirs(settings: Settings, student_id: str) -> int:
    removed = 0
    for path in _student_storage_dirs(settings, student_id):
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
            removed += 1
    return removed


class RateLimiter:
    def __init__(self, max_requests: int, window_sec: float):
        self.max_requests = max(1, max_requests)
        self.window_sec = max(1.0, window_sec)
        self._lock = threading.Lock()
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> tuple[bool, int]:
        now = time.time()
        with self._lock:
            window = self._events[key]
            while window and (now - window[0]) > self.window_sec:
                window.popleft()
            if len(window) >= self.max_requests:
                retry_after = int(max(1.0, self.window_sec - (now - window[0])))
                return False, retry_after
            window.append(now)
            return True, 0


class LiveAttendanceManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False
        self._error = ""
        self._session_id: str | None = None
        self._started_at = 0.0
        self._camera_index = 0
        self._processed_frames = 0
        self._detections = 0
        self._roster_count = 0
        self._present: dict[str, dict[str, Any]] = {}
        self._events: deque[dict[str, Any]] = deque(maxlen=2000)
        self._next_event_id = 0
        self._latest_boxes: list[dict[str, Any]] = []
        self._latest_frame_jpeg: bytes | None = None
        self._csv_path = ""
        self._device = ""

    def _reset_state_locked(self) -> None:
        self._error = ""
        self._session_id = None
        self._started_at = time.time()
        self._processed_frames = 0
        self._detections = 0
        self._roster_count = 0
        self._present = {}
        self._events.clear()
        self._next_event_id = 0
        self._latest_boxes = []
        self._latest_frame_jpeg = None
        self._csv_path = ""

    def _status_locked(self) -> dict[str, Any]:
        elapsed = max(0.0, time.time() - self._started_at) if self._started_at > 0 else 0.0
        return {
            "running": self._running,
            "error": self._error,
            "sessionId": self._session_id,
            "cameraIndex": self._camera_index,
            "device": self._device,
            "elapsedSec": round(elapsed, 2),
            "processedFrames": self._processed_frames,
            "detections": self._detections,
            "presentCount": len(self._present),
            "rosterCount": self._roster_count,
            "lastEventId": self._next_event_id,
            "csvPath": self._csv_path,
            "presentStudents": sorted(
                self._present.values(),
                key=lambda row: str(row.get("name", "")),
            ),
        }

    def status(self) -> dict[str, Any]:
        with self._lock:
            return self._status_locked()

    def events(self, since: int = 0) -> dict[str, Any]:
        with self._lock:
            new_events = [event for event in self._events if int(event["eventId"]) > since]
            return {
                "ok": True,
                "events": new_events,
                "boxes": list(self._latest_boxes),
                "status": self._status_locked(),
            }

    def latest_frame(self) -> bytes | None:
        with self._lock:
            if self._latest_frame_jpeg is None:
                return None
            return bytes(self._latest_frame_jpeg)

    def start(self, camera_index: int = 0, device: str | None = None) -> dict[str, Any]:
        with self._lock:
            if self._running:
                raise RuntimeError("Attendance session is already running.")
            self._running = True
            self._camera_index = camera_index
            self._device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
            self._stop_event.clear()
            self._reset_state_locked()
            self._thread = threading.Thread(
                target=self._run_loop,
                kwargs={
                    "camera_index": camera_index,
                    "device": self._device,
                },
                daemon=True,
            )
            self._thread.start()
            return self._status_locked()

    def stop(self, join_timeout_sec: float = 12.0) -> dict[str, Any]:
        with self._lock:
            thread = self._thread
            running = self._running
            self._stop_event.set()

        if running and thread is not None and thread.is_alive():
            thread.join(timeout=join_timeout_sec)

        with self._lock:
            return self._status_locked()

    @staticmethod
    def _scale_box(
        box: tuple[int, int, int, int],
        frame_resize: float,
        width: int,
        height: int,
    ) -> tuple[int, int, int, int]:
        if abs(frame_resize - 1.0) >= 1e-6 and frame_resize > 0:
            x1, y1, x2, y2 = box
            box = (
                int(x1 / frame_resize),
                int(y1 / frame_resize),
                int(x2 / frame_resize),
                int(y2 / frame_resize),
            )
        x1, y1, x2, y2 = box
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        return x1, y1, x2, y2

    def _draw_overlay(
        self,
        frame,
        matches: list[MatchResult],
        frame_resize: float,
        marked_students: set[str],
    ) -> tuple[Any, list[dict[str, Any]]]:
        overlay = frame.copy()
        height, width = overlay.shape[:2]
        boxes: list[dict[str, Any]] = []
        for match in matches:
            x1, y1, x2, y2 = self._scale_box(match.box, frame_resize, width=width, height=height)
            if x2 <= x1 or y2 <= y1:
                continue

            if match.student_id is None:
                color = (0, 0, 255)
                label = f"Unknown ({match.confidence:.2f})"
                tag = "unknown"
            else:
                already_marked = match.student_id in marked_students
                color = (0, 255, 0) if already_marked else (0, 255, 255)
                label = f"{match.name} ({match.confidence:.2f})"
                tag = "marked" if already_marked else "candidate"

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                overlay,
                label,
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

            boxes.append(
                {
                    "left": round((x1 / width) * 100.0, 3),
                    "top": round((y1 / height) * 100.0, 3),
                    "width": round(((x2 - x1) / width) * 100.0, 3),
                    "height": round(((y2 - y1) / height) * 100.0, 3),
                    "label": label,
                    "tag": tag,
                }
            )
        return overlay, boxes

    def _append_events(self, new_events: list[dict[str, Any]]) -> None:
        if not new_events:
            return
        with self._lock:
            for event in new_events:
                self._next_event_id += 1
                event["eventId"] = self._next_event_id
                self._events.append(event)
                self._present[event["studentId"]] = {
                    "studentId": event["studentId"],
                    "name": event["name"],
                    "confidence": event["confidence"],
                    "detectedAt": event["detectedAt"],
                }

    def _run_loop(self, camera_index: int, device: str) -> None:
        settings = Settings()
        ensure_directories(settings)
        db = AttendanceDB(settings.db_path)
        db.init_schema()
        stream: WebcamStream | None = None
        session_id: str | None = None

        vote_bank: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=max(8, settings.min_votes * 3))
        )
        marked_students: set[str] = set()
        frame_index = 0
        processed_frames = 0
        total_detections = 0
        last_matches: list[MatchResult] = []

        try:
            gallery = PrototypeGallery.from_db(db)
            if gallery.is_empty():
                raise RuntimeError(
                    "No enrolled student prototypes found. Run enrollment and embedding build first."
                )

            engine = FaceRecognitionEngine(
                gallery=gallery,
                device=device,
                image_size=settings.image_size,
                min_face_size=settings.min_face_size,
                detection_probability=settings.detection_probability,
            )
            session_id = db.create_session(mode="live", source=f"camera:{camera_index}")
            roster_count = len(db.list_students())
            stream = WebcamStream(camera_index=camera_index).start()

            with self._lock:
                self._session_id = session_id
                self._roster_count = roster_count

            while not self._stop_event.is_set():
                frame = stream.read()
                if frame is None:
                    time.sleep(0.01)
                    continue

                frame_index += 1
                if frame_index % max(1, settings.sample_every_n_frames) == 0:
                    if settings.frame_resize < 1.0:
                        process_frame = cv2.resize(
                            frame,
                            None,
                            fx=settings.frame_resize,
                            fy=settings.frame_resize,
                            interpolation=cv2.INTER_LINEAR,
                        )
                    else:
                        process_frame = frame

                    last_matches = engine.recognize_frame(
                        process_frame,
                        threshold=settings.recognition_threshold,
                        use_flip_tta=False,
                    )
                    processed_frames += 1
                    total_detections += len(last_matches)

                    new_events: list[dict[str, Any]] = []
                    with db.transaction():
                        for match in last_matches:
                            if match.student_id is None:
                                continue
                            if match.confidence < settings.recognition_threshold:
                                continue

                            votes = vote_bank[match.student_id]
                            votes.append(match.confidence)
                            if len(votes) < settings.min_votes:
                                continue

                            recent_votes = list(votes)[-settings.min_votes :]
                            average_conf = sum(recent_votes) / float(settings.min_votes)
                            if average_conf < settings.mark_threshold:
                                continue
                            if match.student_id in marked_students:
                                continue

                            inserted = db.mark_attendance(
                                session_id=session_id,
                                student_id=match.student_id,
                                confidence=average_conf,
                                frame_index=frame_index,
                            )
                            if inserted:
                                marked_students.add(match.student_id)
                                new_events.append(
                                    {
                                        "studentId": match.student_id,
                                        "name": match.name,
                                        "confidence": round(float(average_conf), 4),
                                        "detectedAt": utc_now_iso(),
                                    }
                                )

                    self._append_events(new_events)

                overlay, boxes = self._draw_overlay(
                    frame=frame,
                    matches=last_matches,
                    frame_resize=settings.frame_resize,
                    marked_students=marked_students,
                )
                cv2.putText(
                    overlay,
                    f"Present: {len(marked_students)}  Processed: {processed_frames}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                ok, encoded = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                frame_bytes = bytes(encoded) if ok else None

                with self._lock:
                    self._processed_frames = processed_frames
                    self._detections = total_detections
                    self._latest_boxes = boxes
                    self._latest_frame_jpeg = frame_bytes

            if session_id is not None:
                csv_path = settings.reports_dir / f"attendance_{session_id}.csv"
                db.export_session_csv(session_id, csv_path)
                with self._lock:
                    self._csv_path = str(csv_path)
        except Exception as exc:
            with self._lock:
                self._error = str(exc)
        finally:
            if stream is not None:
                stream.stop()
            db.close()
            with self._lock:
                self._running = False
                self._thread = None
                self._stop_event.clear()


ATTENDANCE_MANAGER = LiveAttendanceManager()
ENROLL_RATE_LIMITER = RateLimiter(
    max_requests=ENROLL_RATE_LIMIT_MAX,
    window_sec=ENROLL_RATE_LIMIT_WINDOW_SEC,
)


class FrontendHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def _send_json(
        self,
        status: int,
        payload: dict[str, Any],
        headers: dict[str, str] | None = None,
    ) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        if headers:
            for key, value in headers.items():
                self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_jpeg(self, image_bytes: bytes) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(image_bytes)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.end_headers()
        self.wfile.write(image_bytes)

    def _read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            return {}
        raw = self.rfile.read(content_length)
        if not raw:
            return {}
        return json.loads(raw.decode("utf-8"))

    def _client_ip(self) -> str:
        if not self.client_address:
            return "unknown"
        return str(self.client_address[0] or "unknown")

    def _is_loopback_client(self) -> bool:
        return self._client_ip() in {"127.0.0.1", "::1", "localhost"}

    def _require_admin(self) -> bool:
        if ADMIN_TOKEN:
            token = str(self.headers.get("X-Admin-Token", "")).strip()
            if token == ADMIN_TOKEN:
                return True
            self._send_json(
                HTTPStatus.UNAUTHORIZED,
                {"ok": False, "error": "Missing or invalid admin token."},
            )
            return False

        if self._is_loopback_client():
            return True

        self._send_json(
            HTTPStatus.FORBIDDEN,
            {
                "ok": False,
                "error": (
                    "Local admin access only. Set ATTENDANCE_ADMIN_TOKEN for remote admin usage."
                ),
            },
        )
        return False

    def _handle_health(self) -> None:
        self._send_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "service": "frontend-enrollment-attendance-api",
                "attendance": ATTENDANCE_MANAGER.status(),
                "adminTokenConfigured": bool(ADMIN_TOKEN),
            },
        )

    def _handle_list_students(self) -> None:
        settings = Settings()
        ensure_directories(settings)
        db = AttendanceDB(settings.db_path)
        try:
            db.init_schema()
            rows = db.list_students()
            students = [
                {
                    "studentId": str(row["student_id"]),
                    "enrollmentNo": str(row["student_id"]),
                    "name": str(row["name"]),
                    "section": str(row["section"] or ""),
                    "rollNo": str(row["roll_no"] or ""),
                    "createdAt": str(row["created_at"]),
                    "updatedAt": str(row["updated_at"]),
                }
                for row in rows
            ]
        finally:
            db.close()

        self._send_json(HTTPStatus.OK, {"ok": True, "students": students})

    def _parse_student_payload(self, payload: dict[str, Any]) -> tuple[str, str, str, str]:
        enrollment_no = payload.get("enrollmentNo", payload.get("studentId", ""))
        student_id = str(enrollment_no).strip()
        name = str(payload.get("name", "")).strip()
        section = str(payload.get("section", "")).strip()
        roll_no = str(payload.get("rollNo", payload.get("roll_no", ""))).strip()
        return student_id, name, section, roll_no

    def _handle_upsert_student(self) -> None:
        if not self._require_admin():
            return

        try:
            payload = self._read_json()
        except json.JSONDecodeError:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": "Invalid JSON payload."},
            )
            return

        student_id, name, section, roll_no = self._parse_student_payload(payload)
        old_student_id = str(payload.get("oldStudentId", "")).strip()
        if not old_student_id:
            old_student_id = student_id
        if not student_id or not name:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": "enrollmentNo and name are required."},
            )
            return

        settings = Settings()
        ensure_directories(settings)
        db = AttendanceDB(settings.db_path)
        try:
            db.init_schema()
            try:
                updated = db.rename_student(
                    old_student_id=old_student_id,
                    new_student_id=student_id,
                    name=name,
                    section=section,
                    roll_no=roll_no,
                )
                if not updated:
                    db.upsert_student(
                        student_id=student_id,
                        name=name,
                        section=section,
                        roll_no=roll_no,
                    )
                elif old_student_id != student_id:
                    _rename_student_storage_dirs(settings, old_student_id, student_id)
            except ValueError as exc:
                self._send_json(
                    HTTPStatus.CONFLICT,
                    {"ok": False, "error": str(exc)},
                )
                return
            row = db.get_student(student_id)
        finally:
            db.close()

        if row is None:
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"ok": False, "error": "Failed to save student."},
            )
            return

        self._send_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "student": {
                    "studentId": str(row["student_id"]),
                    "enrollmentNo": str(row["student_id"]),
                    "name": str(row["name"]),
                    "section": str(row["section"] or ""),
                    "rollNo": str(row["roll_no"] or ""),
                    "createdAt": str(row["created_at"]),
                    "updatedAt": str(row["updated_at"]),
                },
            },
        )

    def _handle_delete_student(self, student_id: str) -> None:
        if not self._require_admin():
            return

        sid = student_id.strip()
        if not sid:
            self._send_json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": "studentId is required."})
            return

        settings = Settings()
        ensure_directories(settings)
        db = AttendanceDB(settings.db_path)
        try:
            db.init_schema()
            deleted = db.delete_student(sid)
        finally:
            db.close()

        if not deleted:
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"ok": False, "error": f"Student not found: {sid}"},
            )
            return

        removed_dirs = _delete_student_storage_dirs(settings, sid)
        self._send_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "deletedStudentId": sid,
                "removedStorageDirs": removed_dirs,
            },
        )

    def _handle_clear_students(self) -> None:
        if not self._require_admin():
            return

        settings = Settings()
        ensure_directories(settings)
        db = AttendanceDB(settings.db_path)
        try:
            db.init_schema()
            student_ids = [str(row["student_id"]) for row in db.list_students()]
            deleted_rows = db.clear_students()
        finally:
            db.close()

        removed_dirs = 0
        for sid in student_ids:
            removed_dirs += _delete_student_storage_dirs(settings, sid)

        self._send_json(
            HTTPStatus.OK,
            {"ok": True, "deletedRows": deleted_rows, "removedStorageDirs": removed_dirs},
        )

    def _handle_enroll_webcam(self) -> None:
        if not self._require_admin():
            return

        allowed, retry_after = ENROLL_RATE_LIMITER.allow(key=f"enroll:{self._client_ip()}")
        if not allowed:
            self._send_json(
                HTTPStatus.TOO_MANY_REQUESTS,
                {
                    "ok": False,
                    "error": "Rate limit exceeded for enrollment endpoint.",
                    "retryAfterSec": retry_after,
                },
                headers={"Retry-After": str(retry_after)},
            )
            return

        try:
            payload = self._read_json()
        except json.JSONDecodeError:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": "Invalid JSON payload."},
            )
            return

        enrollment_no = payload.get("enrollmentNo", payload.get("studentId", ""))
        student_id = str(enrollment_no).strip()
        old_student_id = str(payload.get("oldStudentId", "")).strip() or student_id
        name = str(payload.get("name", "")).strip()
        section = str(payload.get("section", "")).strip()
        roll_no = str(payload.get("rollNo", payload.get("roll_no", ""))).strip()
        if not student_id or not name:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": "enrollmentNo and name are required."},
            )
            return

        try:
            num_images = int(payload.get("numImages", 25))
            camera_index = int(payload.get("cameraIndex", 0))
            capture_interval = float(payload.get("captureInterval", 0.25))
            min_probability = float(payload.get("minProbability", 0.90))
            camera_width = int(payload.get("cameraWidth", 640))
            camera_height = int(payload.get("cameraHeight", 480))
            build_embeddings = bool(payload.get("buildEmbeddings", True))
            requested_device = str(payload.get("device", "")).strip()
        except (TypeError, ValueError):
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": "Invalid numeric enrollment settings."},
            )
            return

        if num_images < 1 or num_images > 300:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": "numImages must be between 1 and 300."},
            )
            return
        if capture_interval <= 0:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": "captureInterval must be greater than 0."},
            )
            return
        if min_probability < 0.5 or min_probability > 0.999:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": "minProbability must be between 0.5 and 0.999."},
            )
            return
        if camera_width < 320 or camera_width > 1920:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": "cameraWidth must be between 320 and 1920."},
            )
            return
        if camera_height < 240 or camera_height > 1080:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": "cameraHeight must be between 240 and 1080."},
            )
            return

        if not ENROLL_LOCK.acquire(blocking=False):
            self._send_json(
                HTTPStatus.CONFLICT,
                {
                    "ok": False,
                    "error": "Another enrollment session is already running.",
                },
            )
            return

        device = requested_device or ("cuda:0" if torch.cuda.is_available() else "cpu")

        settings = Settings()
        ensure_directories(settings)
        db = AttendanceDB(settings.db_path)
        embeddings_built = 0
        try:
            db.init_schema()
            try:
                updated = db.rename_student(
                    old_student_id=old_student_id,
                    new_student_id=student_id,
                    name=name,
                    section=section,
                    roll_no=roll_no,
                )
                if not updated:
                    db.upsert_student(
                        student_id=student_id,
                        name=name,
                        section=section,
                        roll_no=roll_no,
                    )
                elif old_student_id != student_id:
                    _rename_student_storage_dirs(settings, old_student_id, student_id)
            except ValueError as exc:
                self._send_json(
                    HTTPStatus.CONFLICT,
                    {"ok": False, "error": str(exc)},
                )
                return
            saved = capture_student_from_webcam(
                db=db,
                settings=settings,
                student_id=student_id,
                name=name,
                num_images=num_images,
                camera_index=camera_index,
                capture_interval=capture_interval,
                min_probability=min_probability,
                camera_width=camera_width,
                camera_height=camera_height,
                device=device,
            )
            if saved > 0 and build_embeddings:
                builder = EmbeddingBuilder(db=db, settings=settings, device=device)
                embeddings_built = builder.build_for_student(student_id)
        except Exception as exc:
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"ok": False, "error": f"Enrollment failed: {exc}"},
            )
            return
        finally:
            db.close()
            ENROLL_LOCK.release()

        self._send_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "studentId": student_id,
                "enrollmentNo": student_id,
                "name": name,
                "section": section,
                "rollNo": roll_no,
                "saved": saved,
                "requested": num_images,
                "embeddingsBuilt": embeddings_built,
            },
        )

    def _handle_attendance_start(self) -> None:
        if not self._require_admin():
            return

        try:
            payload = self._read_json()
        except json.JSONDecodeError:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": "Invalid JSON payload."},
            )
            return

        try:
            camera_index = int(payload.get("cameraIndex", 0))
        except (TypeError, ValueError):
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": "cameraIndex must be an integer."},
            )
            return

        requested_device = str(payload.get("device", "")).strip() or None
        try:
            status = ATTENDANCE_MANAGER.start(camera_index=camera_index, device=requested_device)
        except RuntimeError as exc:
            self._send_json(HTTPStatus.CONFLICT, {"ok": False, "error": str(exc)})
            return

        self._send_json(HTTPStatus.OK, {"ok": True, "attendance": status})

    def _handle_attendance_stop(self) -> None:
        if not self._require_admin():
            return
        status = ATTENDANCE_MANAGER.stop()
        self._send_json(HTTPStatus.OK, {"ok": True, "attendance": status})

    def _handle_attendance_status(self) -> None:
        self._send_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "attendance": ATTENDANCE_MANAGER.status(),
            },
        )

    def _handle_attendance_events(self, query: dict[str, list[str]]) -> None:
        since_value = query.get("since", ["0"])[0]
        try:
            since = max(0, int(since_value))
        except (TypeError, ValueError):
            since = 0
        payload = ATTENDANCE_MANAGER.events(since=since)
        self._send_json(HTTPStatus.OK, payload)

    def _handle_attendance_frame(self) -> None:
        frame = ATTENDANCE_MANAGER.latest_frame()
        if frame is None:
            self._send_json(
                HTTPStatus.NOT_FOUND,
                {"ok": False, "error": "No attendance frame available yet."},
            )
            return
        self._send_jpeg(frame)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        if path == "/api/health":
            self._handle_health()
            return
        if path == "/api/students":
            self._handle_list_students()
            return
        if path == "/api/attendance/status":
            self._handle_attendance_status()
            return
        if path == "/api/attendance/events":
            self._handle_attendance_events(query=query)
            return
        if path == "/api/attendance/frame":
            self._handle_attendance_frame()
            return
        super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/enroll/webcam":
            self._handle_enroll_webcam()
            return
        if path == "/api/students":
            self._handle_upsert_student()
            return
        if path == "/api/attendance/start":
            self._handle_attendance_start()
            return
        if path == "/api/attendance/stop":
            self._handle_attendance_stop()
            return
        self._send_json(
            HTTPStatus.NOT_FOUND,
            {"ok": False, "error": f"Route not found: {path}"},
        )

    def do_DELETE(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/students":
            self._handle_clear_students()
            return

        prefix = "/api/students/"
        if path.startswith(prefix):
            student_id = unquote(path[len(prefix) :]).strip()
            self._handle_delete_student(student_id=student_id)
            return

        self._send_json(
            HTTPStatus.NOT_FOUND,
            {"ok": False, "error": f"Route not found: {path}"},
        )


def main() -> None:
    args = parse_args()
    if not FRONTEND_DIR.exists():
        raise SystemExit(f"Frontend folder not found: {FRONTEND_DIR}")

    server = ThreadingHTTPServer((args.host, args.port), FrontendHandler)
    print(f"Frontend server running at http://{args.host}:{args.port}")
    print("Open /enrollment.html for student enrollment.")
    print("Open /index.html for live attendance dashboard.")
    if ADMIN_TOKEN:
        print("Admin token auth is enabled (header: X-Admin-Token).")
    else:
        print("Admin token auth is not set; admin APIs are restricted to loopback clients.")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        ATTENDANCE_MANAGER.stop(join_timeout_sec=2.0)
        server.server_close()


if __name__ == "__main__":
    main()
