from __future__ import annotations

from pathlib import Path

from .db import AttendanceDB


def format_session_summary(db: AttendanceDB, session_id: str) -> str:
    session_rows = [row for row in db.list_sessions(limit=200) if row["session_id"] == session_id]
    if not session_rows:
        return f"Session not found: {session_id}"

    session = session_rows[0]
    records = db.get_session_records(session_id)
    lines = [
        f"Session ID : {session['session_id']}",
        f"Mode       : {session['mode']}",
        f"Source     : {session['source']}",
        f"Started    : {session['started_at']}",
        f"Present    : {len(records)} students",
        "-" * 60,
    ]
    if not records:
        lines.append("No attendance records in this session.")
        return "\n".join(lines)

    lines.append(f"{'Student ID':<20} {'Name':<24} {'Confidence':>10}")
    lines.append("-" * 60)
    for row in records:
        lines.append(
            f"{row['student_id']:<20} {row['name']:<24} {float(row['confidence']):>10.3f}"
        )
    return "\n".join(lines)


def export_latest_session(db: AttendanceDB, output_path: Path) -> Path:
    latest = db.latest_session_id()
    if latest is None:
        raise RuntimeError("No attendance sessions found.")
    return db.export_session_csv(latest, output_path)

