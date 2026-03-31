from __future__ import annotations

import csv
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np

from .utils import utc_now_iso


class AttendanceDB:
    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON;")
        self._tx_depth = 0

    def close(self) -> None:
        self.conn.close()

    def init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS students (
                student_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                section TEXT NOT NULL DEFAULT '',
                roll_no TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                vector BLOB NOT NULL,
                dim INTEGER NOT NULL,
                source_path TEXT,
                quality REAL DEFAULT 1.0,
                created_at TEXT NOT NULL,
                FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_embeddings_student_id
            ON embeddings(student_id);

            CREATE TABLE IF NOT EXISTS student_prototypes (
                student_id TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                dim INTEGER NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS attendance_sessions (
                session_id TEXT PRIMARY KEY,
                mode TEXT NOT NULL,
                source TEXT,
                started_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS attendance_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                student_id TEXT NOT NULL,
                confidence REAL NOT NULL,
                frame_index INTEGER,
                timestamp TEXT NOT NULL,
                UNIQUE(session_id, student_id),
                FOREIGN KEY (session_id) REFERENCES attendance_sessions(session_id) ON DELETE CASCADE,
                FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_records_session_id
            ON attendance_records(session_id);
            """
        )
        self._ensure_student_columns()
        self._commit_if_needed()

    def _ensure_student_columns(self) -> None:
        existing = {
            str(row["name"]).lower()
            for row in self.conn.execute("PRAGMA table_info(students)").fetchall()
        }
        if "section" not in existing:
            self.conn.execute("ALTER TABLE students ADD COLUMN section TEXT NOT NULL DEFAULT ''")
        if "roll_no" not in existing:
            self.conn.execute("ALTER TABLE students ADD COLUMN roll_no TEXT NOT NULL DEFAULT ''")

    def _commit_if_needed(self) -> None:
        if self._tx_depth == 0:
            self.conn.commit()

    @contextmanager
    def transaction(self):
        self._tx_depth += 1
        try:
            yield
        except Exception:
            self._tx_depth = 0
            self.conn.rollback()
            raise
        else:
            self._tx_depth -= 1
            if self._tx_depth == 0:
                self.conn.commit()

    @staticmethod
    def _vec_to_blob(vector: np.ndarray) -> bytes:
        return np.asarray(vector, dtype=np.float32).tobytes()

    @staticmethod
    def _blob_to_vec(blob: bytes, dim: int) -> np.ndarray:
        arr = np.frombuffer(blob, dtype=np.float32)
        if arr.shape[0] != dim:
            raise ValueError(f"Embedding dim mismatch: expected {dim}, got {arr.shape[0]}")
        return arr.copy()

    def upsert_student(
        self,
        student_id: str,
        name: str,
        section: str | None = None,
        roll_no: str | None = None,
    ) -> None:
        if section is None or roll_no is None:
            existing = self.get_student(student_id)
            if existing is not None:
                if section is None:
                    section = str(existing["section"] or "")
                if roll_no is None:
                    roll_no = str(existing["roll_no"] or "")
        section_value = "" if section is None else section
        roll_no_value = "" if roll_no is None else roll_no

        now = utc_now_iso()
        self.conn.execute(
            """
            INSERT INTO students(student_id, name, section, roll_no, created_at, updated_at)
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(student_id) DO UPDATE SET
                name = excluded.name,
                section = excluded.section,
                roll_no = excluded.roll_no,
                updated_at = excluded.updated_at
            """,
            (student_id, name, section_value, roll_no_value, now, now),
        )
        self._commit_if_needed()

    def list_students(self) -> list[sqlite3.Row]:
        rows = self.conn.execute(
            """
            SELECT student_id, name, section, roll_no, created_at, updated_at
            FROM students
            ORDER BY name ASC
            """
        ).fetchall()
        return rows

    def get_student(self, student_id: str) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            """
            SELECT student_id, name, section, roll_no, created_at, updated_at
            FROM students
            WHERE student_id = ?
            """,
            (student_id,),
        ).fetchone()

    def delete_student(self, student_id: str) -> bool:
        cur = self.conn.execute("DELETE FROM students WHERE student_id = ?", (student_id,))
        self._commit_if_needed()
        return cur.rowcount > 0

    def clear_students(self) -> int:
        cur = self.conn.execute("DELETE FROM students")
        self._commit_if_needed()
        return int(cur.rowcount)

    def rename_student(
        self,
        old_student_id: str,
        new_student_id: str,
        name: str | None = None,
        section: str | None = None,
        roll_no: str | None = None,
    ) -> bool:
        old_row = self.get_student(old_student_id)
        if old_row is None:
            return False

        if old_student_id == new_student_id:
            self.upsert_student(
                student_id=new_student_id,
                name=name if name is not None else str(old_row["name"]),
                section=section if section is not None else str(old_row["section"] or ""),
                roll_no=roll_no if roll_no is not None else str(old_row["roll_no"] or ""),
            )
            return True

        if self.get_student(new_student_id) is not None:
            raise ValueError(f"Student ID already exists: {new_student_id}")

        new_name = name if name is not None else str(old_row["name"])
        new_section = section if section is not None else str(old_row["section"] or "")
        new_roll_no = roll_no if roll_no is not None else str(old_row["roll_no"] or "")
        now = utc_now_iso()
        created_at = str(old_row["created_at"])

        with self.transaction():
            self.conn.execute(
                """
                INSERT INTO students(student_id, name, section, roll_no, created_at, updated_at)
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (new_student_id, new_name, new_section, new_roll_no, created_at, now),
            )
            self.conn.execute(
                "UPDATE embeddings SET student_id = ? WHERE student_id = ?",
                (new_student_id, old_student_id),
            )
            self.conn.execute(
                "UPDATE student_prototypes SET student_id = ? WHERE student_id = ?",
                (new_student_id, old_student_id),
            )
            self.conn.execute(
                "UPDATE attendance_records SET student_id = ? WHERE student_id = ?",
                (new_student_id, old_student_id),
            )
            self.conn.execute("DELETE FROM students WHERE student_id = ?", (old_student_id,))
        return True

    def clear_embeddings(self, student_id: Optional[str] = None) -> None:
        if student_id:
            self.conn.execute("DELETE FROM embeddings WHERE student_id = ?", (student_id,))
            self.conn.execute("DELETE FROM student_prototypes WHERE student_id = ?", (student_id,))
        else:
            self.conn.execute("DELETE FROM embeddings")
            self.conn.execute("DELETE FROM student_prototypes")
        self._commit_if_needed()

    def add_embedding(
        self,
        student_id: str,
        embedding: np.ndarray,
        source_path: str = "",
        quality: float = 1.0,
    ) -> None:
        vector = np.asarray(embedding, dtype=np.float32)
        now = utc_now_iso()
        self.conn.execute(
            """
            INSERT INTO embeddings(student_id, vector, dim, source_path, quality, created_at)
            VALUES(?, ?, ?, ?, ?, ?)
            """,
            (
                student_id,
                self._vec_to_blob(vector),
                int(vector.shape[0]),
                source_path,
                float(quality),
                now,
            ),
        )
        self._commit_if_needed()

    def set_prototype(self, student_id: str, embedding: np.ndarray) -> None:
        vector = np.asarray(embedding, dtype=np.float32)
        now = utc_now_iso()
        self.conn.execute(
            """
            INSERT INTO student_prototypes(student_id, vector, dim, updated_at)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(student_id) DO UPDATE SET
                vector = excluded.vector,
                dim = excluded.dim,
                updated_at = excluded.updated_at
            """,
            (student_id, self._vec_to_blob(vector), int(vector.shape[0]), now),
        )
        self._commit_if_needed()

    def load_prototypes(self) -> tuple[list[str], list[str], np.ndarray]:
        rows = self.conn.execute(
            """
            SELECT p.student_id, s.name, p.vector, p.dim
            FROM student_prototypes p
            JOIN students s ON s.student_id = p.student_id
            ORDER BY s.name ASC
            """
        ).fetchall()
        if not rows:
            return [], [], np.empty((0, 0), dtype=np.float32)

        student_ids: list[str] = []
        names: list[str] = []
        matrix: list[np.ndarray] = []
        for row in rows:
            student_ids.append(row["student_id"])
            names.append(row["name"])
            matrix.append(self._blob_to_vec(row["vector"], row["dim"]))
        return student_ids, names, np.vstack(matrix).astype(np.float32)

    def create_session(self, mode: str, source: str = "") -> str:
        session_id = uuid.uuid4().hex
        self.conn.execute(
            """
            INSERT INTO attendance_sessions(session_id, mode, source, started_at)
            VALUES(?, ?, ?, ?)
            """,
            (session_id, mode, source, utc_now_iso()),
        )
        self._commit_if_needed()
        return session_id

    def mark_attendance(
        self,
        session_id: str,
        student_id: str,
        confidence: float,
        frame_index: Optional[int] = None,
    ) -> bool:
        cur = self.conn.execute(
            """
            INSERT OR IGNORE INTO attendance_records(
                session_id, student_id, confidence, frame_index, timestamp
            )
            VALUES(?, ?, ?, ?, ?)
            """,
            (session_id, student_id, float(confidence), frame_index, utc_now_iso()),
        )
        self._commit_if_needed()
        return cur.rowcount > 0

    def get_session_records(self, session_id: str) -> list[sqlite3.Row]:
        rows = self.conn.execute(
            """
            SELECT
                r.session_id,
                r.student_id,
                s.name,
                r.confidence,
                r.frame_index,
                r.timestamp
            FROM attendance_records r
            JOIN students s ON s.student_id = r.student_id
            WHERE r.session_id = ?
            ORDER BY s.name ASC
            """,
            (session_id,),
        ).fetchall()
        return rows

    def list_sessions(self, limit: int = 20) -> list[sqlite3.Row]:
        rows = self.conn.execute(
            """
            SELECT
                s.session_id,
                s.mode,
                s.source,
                s.started_at,
                COUNT(r.id) AS present_count
            FROM attendance_sessions s
            LEFT JOIN attendance_records r ON r.session_id = s.session_id
            GROUP BY s.session_id
            ORDER BY s.started_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return rows

    def latest_session_id(self) -> Optional[str]:
        row = self.conn.execute(
            "SELECT session_id FROM attendance_sessions ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        return str(row["session_id"])

    def export_session_csv(self, session_id: str, output_csv_path: Path | str) -> Path:
        output_path = Path(output_csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        session = self.conn.execute(
            "SELECT session_id, mode, source, started_at FROM attendance_sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if session is None:
            raise ValueError(f"Session not found: {session_id}")

        rows = self.get_session_records(session_id)
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "session_id",
                    "mode",
                    "source",
                    "session_started_at",
                    "student_id",
                    "name",
                    "confidence",
                    "frame_index",
                    "timestamp",
                ]
            )
            for row in rows:
                writer.writerow(
                    [
                        session["session_id"],
                        session["mode"],
                        session["source"],
                        session["started_at"],
                        row["student_id"],
                        row["name"],
                        f"{row['confidence']:.4f}",
                        row["frame_index"],
                        row["timestamp"],
                    ]
                )
        return output_path
