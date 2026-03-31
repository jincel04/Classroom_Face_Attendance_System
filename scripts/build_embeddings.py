import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attendance_system.config import Settings, ensure_directories
from attendance_system.db import AttendanceDB
from attendance_system.embedding_builder import EmbeddingBuilder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or rebuild face embeddings for enrolled students.")
    parser.add_argument(
        "--student-id",
        default=None,
        help="Build embeddings for one student ID. Omit to rebuild all students.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='Inference device (e.g. "cpu" or "cuda:0"). Auto-detect when omitted.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    settings = Settings()
    ensure_directories(settings)
    db = AttendanceDB(settings.db_path)
    db.init_schema()

    try:
        builder = EmbeddingBuilder(db=db, settings=settings, device=device)
        results = builder.build_all(student_id=args.student_id)
        total_students = len(results)
        total_embeddings = sum(results.values())
        print(f"Embedding build complete on {device}")
        print(f"Students processed: {total_students}")
        print(f"Embeddings generated: {total_embeddings}")
        for student_id, count in sorted(results.items()):
            print(f"  {student_id}: {count}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
