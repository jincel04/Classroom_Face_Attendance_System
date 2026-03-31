import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attendance_system.config import Settings, ensure_directories
from attendance_system.db import AttendanceDB


def main() -> None:
    settings = Settings()
    ensure_directories(settings)
    db = AttendanceDB(settings.db_path)
    try:
        db.init_schema()
        print(f"Database initialized: {settings.db_path}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
