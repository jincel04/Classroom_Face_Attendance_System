from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .camera import WebcamStream
from .config import Settings
from .db import AttendanceDB
from .recognition import FaceRecognitionEngine, MatchResult


@dataclass
class AttendanceResult:
    session_id: str
    mode: str
    elapsed_sec: float
    present_count: int
    processed_frames: int
    detections: int
    csv_path: Path
    annotated_image_path: Path | None = None


class AttendanceRunner:
    def __init__(self, db: AttendanceDB, engine: FaceRecognitionEngine, settings: Settings):
        self.db = db
        self.engine = engine
        self.settings = settings

    def _scale_box(self, box: tuple[int, int, int, int], frame_resize: float) -> tuple[int, int, int, int]:
        if abs(frame_resize - 1.0) < 1e-6:
            return box
        x1, y1, x2, y2 = box
        return (
            int(x1 / frame_resize),
            int(y1 / frame_resize),
            int(x2 / frame_resize),
            int(y2 / frame_resize),
        )

    def _draw_matches(
        self,
        frame: np.ndarray,
        matches: list[MatchResult],
        frame_resize: float,
        marked_students: set[str],
        elapsed_sec: float,
    ) -> np.ndarray:
        display = frame.copy()
        for match in matches:
            x1, y1, x2, y2 = self._scale_box(match.box, frame_resize)
            if match.student_id is None:
                color = (0, 0, 255)
                label = f"Unknown ({match.confidence:.2f})"
            else:
                color = (0, 255, 0) if match.student_id in marked_students else (0, 255, 255)
                label = f"{match.name} ({match.confidence:.2f})"

            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                display,
                label,
                (x1, max(24, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            display,
            f"Elapsed: {elapsed_sec:.1f}s  Present: {len(marked_students)}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return display

    def _mark_from_matches(
        self,
        matches: list[MatchResult],
        session_id: str,
        frame_index: int,
        vote_bank: dict[str, deque[float]],
        marked_students: set[str],
    ) -> int:
        newly_marked = 0
        with self.db.transaction():
            for match in matches:
                if match.student_id is None:
                    continue
                if match.confidence < self.settings.recognition_threshold:
                    continue

                votes = vote_bank[match.student_id]
                votes.append(match.confidence)
                if len(votes) < self.settings.min_votes:
                    continue

                recent_votes = list(votes)[-self.settings.min_votes :]
                average_conf = float(np.mean(recent_votes))
                if average_conf < self.settings.mark_threshold:
                    continue
                if match.student_id in marked_students:
                    continue

                inserted = self.db.mark_attendance(
                    session_id=session_id,
                    student_id=match.student_id,
                    confidence=average_conf,
                    frame_index=frame_index,
                )
                if inserted:
                    marked_students.add(match.student_id)
                    newly_marked += 1
        return newly_marked

    def run_live(self, camera_index: int = 0, duration_sec: int | None = None, display: bool = True) -> AttendanceResult:
        if duration_sec is not None:
            duration = max(0, duration_sec)
        elif self.settings.session_duration_sec > 0:
            duration = self.settings.session_duration_sec
        else:
            duration = None

        idle_timeout = self.settings.idle_timeout_sec if self.settings.idle_timeout_sec > 0 else None
        session_id = self.db.create_session(mode="live", source=f"camera:{camera_index}")
        stream = WebcamStream(camera_index=camera_index).start()

        vote_bank: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=max(8, self.settings.min_votes * 3)))
        marked_students: set[str] = set()

        frame_index = 0
        processed_frames = 0
        total_detections = 0
        last_matches: list[MatchResult] = []
        start = time.time()
        last_new_mark = start

        try:
            while True:
                now = time.time()
                elapsed = now - start
                if duration is not None and elapsed >= duration:
                    break

                frame = stream.read()
                if frame is None:
                    continue

                frame_index += 1
                should_process = frame_index % self.settings.sample_every_n_frames == 0
                if should_process:
                    if self.settings.frame_resize < 1.0:
                        process_frame = cv2.resize(
                            frame,
                            None,
                            fx=self.settings.frame_resize,
                            fy=self.settings.frame_resize,
                            interpolation=cv2.INTER_LINEAR,
                        )
                    else:
                        process_frame = frame

                    last_matches = self.engine.recognize_frame(
                        process_frame,
                        threshold=self.settings.recognition_threshold,
                        use_flip_tta=False,
                    )
                    total_detections += len(last_matches)
                    processed_frames += 1

                    newly_marked = self._mark_from_matches(
                        matches=last_matches,
                        session_id=session_id,
                        frame_index=frame_index,
                        vote_bank=vote_bank,
                        marked_students=marked_students,
                    )
                    if newly_marked > 0:
                        last_new_mark = now

                if display:
                    overlay = self._draw_matches(
                        frame=frame,
                        matches=last_matches,
                        frame_resize=self.settings.frame_resize,
                        marked_students=marked_students,
                        elapsed_sec=elapsed,
                    )
                    cv2.imshow("Attendance Session - press Q to quit", overlay)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break

                if idle_timeout is not None and elapsed > 15 and (now - last_new_mark) > idle_timeout:
                    break
        finally:
            stream.stop()
            if display:
                cv2.destroyAllWindows()

        elapsed_total = time.time() - start
        csv_path = self.settings.reports_dir / f"attendance_{session_id}.csv"
        self.db.export_session_csv(session_id, csv_path)

        return AttendanceResult(
            session_id=session_id,
            mode="live",
            elapsed_sec=elapsed_total,
            present_count=len(marked_students),
            processed_frames=processed_frames,
            detections=total_detections,
            csv_path=csv_path,
            annotated_image_path=None,
        )

    def run_video_file(
        self,
        video_path: Path,
        duration_sec: int | None = None,
        display: bool = False,
    ) -> AttendanceResult:
        if duration_sec is not None:
            duration = max(0, duration_sec)
        elif self.settings.session_duration_sec > 0:
            duration = self.settings.session_duration_sec
        else:
            duration = None

        idle_timeout = self.settings.idle_timeout_sec if self.settings.idle_timeout_sec > 0 else None
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video file: {video_path}")

        session_id = self.db.create_session(mode="video", source=str(video_path))
        vote_bank: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=max(8, self.settings.min_votes * 3)))
        marked_students: set[str] = set()
        frame_index = 0
        processed_frames = 0
        total_detections = 0
        last_matches: list[MatchResult] = []
        start = time.time()
        last_new_mark = start

        try:
            while True:
                now = time.time()
                elapsed = now - start
                if duration is not None and elapsed >= duration:
                    break

                ok, frame = capture.read()
                if not ok:
                    break

                frame_index += 1
                should_process = frame_index % self.settings.sample_every_n_frames == 0
                if should_process:
                    if self.settings.frame_resize < 1.0:
                        process_frame = cv2.resize(
                            frame,
                            None,
                            fx=self.settings.frame_resize,
                            fy=self.settings.frame_resize,
                            interpolation=cv2.INTER_LINEAR,
                        )
                    else:
                        process_frame = frame

                    last_matches = self.engine.recognize_frame(
                        process_frame,
                        threshold=self.settings.recognition_threshold,
                        use_flip_tta=False,
                    )
                    total_detections += len(last_matches)
                    processed_frames += 1

                    newly_marked = self._mark_from_matches(
                        matches=last_matches,
                        session_id=session_id,
                        frame_index=frame_index,
                        vote_bank=vote_bank,
                        marked_students=marked_students,
                    )
                    if newly_marked > 0:
                        last_new_mark = now

                if display:
                    overlay = self._draw_matches(
                        frame=frame,
                        matches=last_matches,
                        frame_resize=self.settings.frame_resize,
                        marked_students=marked_students,
                        elapsed_sec=elapsed,
                    )
                    cv2.imshow("Attendance Video - press Q to quit", overlay)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break

                if idle_timeout is not None and elapsed > 15 and (now - last_new_mark) > idle_timeout:
                    break
        finally:
            capture.release()
            if display:
                cv2.destroyAllWindows()

        elapsed_total = time.time() - start
        csv_path = self.settings.reports_dir / f"attendance_{session_id}.csv"
        self.db.export_session_csv(session_id, csv_path)

        return AttendanceResult(
            session_id=session_id,
            mode="video",
            elapsed_sec=elapsed_total,
            present_count=len(marked_students),
            processed_frames=processed_frames,
            detections=total_detections,
            csv_path=csv_path,
            annotated_image_path=None,
        )

    def run_group_image(self, image_path: Path, display: bool = False) -> AttendanceResult:
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Unable to read image: {image_path}")

        max_width = 1920
        resize_factor = 1.0
        if frame.shape[1] > max_width:
            resize_factor = max_width / frame.shape[1]
            frame = cv2.resize(
                frame,
                None,
                fx=resize_factor,
                fy=resize_factor,
                interpolation=cv2.INTER_LINEAR,
            )

        session_id = self.db.create_session(mode="image", source=str(image_path))
        start = time.time()

        matches = self.engine.recognize_frame(
            frame,
            threshold=self.settings.mark_threshold,
            use_flip_tta=True,
        )
        marked_students: set[str] = set()
        for match in matches:
            if match.student_id is None:
                continue
            if match.confidence < self.settings.mark_threshold:
                continue
            inserted = self.db.mark_attendance(
                session_id=session_id,
                student_id=match.student_id,
                confidence=match.confidence,
                frame_index=0,
            )
            if inserted:
                marked_students.add(match.student_id)

        elapsed_total = time.time() - start
        csv_path = self.settings.reports_dir / f"attendance_{session_id}.csv"
        self.db.export_session_csv(session_id, csv_path)

        annotated = self._draw_matches(
            frame=frame,
            matches=matches,
            frame_resize=1.0,
            marked_students=marked_students,
            elapsed_sec=elapsed_total,
        )
        annotated_path = self.settings.reports_dir / f"annotated_{session_id}.jpg"
        cv2.imwrite(str(annotated_path), annotated)

        if display:
            cv2.imshow("Attendance Group Photo - press any key", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return AttendanceResult(
            session_id=session_id,
            mode="image",
            elapsed_sec=elapsed_total,
            present_count=len(marked_students),
            processed_frames=1,
            detections=len(matches),
            csv_path=csv_path,
            annotated_image_path=annotated_path,
        )
