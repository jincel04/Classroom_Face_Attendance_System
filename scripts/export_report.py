import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attendance_system.config import Settings, ensure_directories
from attendance_system.db import AttendanceDB
from attendance_system.report import format_session_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export attendance session to CSV.")
    parser.add_argument(
        "--session-id",
        default=None,
        help="Session ID to export. Omit to export the latest session.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output path. Defaults to data/reports/attendance_<session>.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings()
    ensure_directories(settings)
    db = AttendanceDB(settings.db_path)
    db.init_schema()

    try:
        session_id = args.session_id or db.latest_session_id()
        if session_id is None:
            raise SystemExit("No attendance sessions found.")
        output_csv = args.output_csv or settings.reports_dir / f"attendance_{session_id}.csv"
        path = db.export_session_csv(session_id, output_csv)
        print(f"Exported: {path}")
        print()
        print(format_session_summary(db, session_id))
    finally:
        db.close()


if __name__ == "__main__":
    main()
