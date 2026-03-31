import numpy as np

from attendance_system.db import AttendanceDB
from attendance_system.utils import l2_normalize


def test_db_attendance_deduplicates_per_session(tmp_path):
    db_path = tmp_path / "attendance_test.db"
    db = AttendanceDB(db_path)
    db.init_schema()

    try:
        db.upsert_student("S001", "Ada Lovelace")
        emb = l2_normalize(np.random.rand(512).astype(np.float32))
        db.add_embedding("S001", emb, source_path="sample.jpg")
        db.set_prototype("S001", emb)

        session_id = db.create_session(mode="live", source="camera:0")
        first_insert = db.mark_attendance(session_id, "S001", confidence=0.88, frame_index=10)
        second_insert = db.mark_attendance(session_id, "S001", confidence=0.92, frame_index=20)
        rows = db.get_session_records(session_id)

        assert first_insert is True
        assert second_insert is False
        assert len(rows) == 1
        assert rows[0]["student_id"] == "S001"
    finally:
        db.close()


def test_db_load_prototypes(tmp_path):
    db_path = tmp_path / "attendance_test.db"
    db = AttendanceDB(db_path)
    db.init_schema()
    try:
        db.upsert_student("S001", "Ada Lovelace")
        db.upsert_student("S002", "Alan Turing")
        emb1 = l2_normalize(np.random.rand(512).astype(np.float32))
        emb2 = l2_normalize(np.random.rand(512).astype(np.float32))
        db.set_prototype("S001", emb1)
        db.set_prototype("S002", emb2)

        student_ids, names, matrix = db.load_prototypes()
        assert student_ids == ["S001", "S002"]
        assert names == ["Ada Lovelace", "Alan Turing"]
        assert matrix.shape == (2, 512)
    finally:
        db.close()


def test_db_upsert_student_preserves_metadata_when_optional_fields_omitted(tmp_path):
    db_path = tmp_path / "attendance_test.db"
    db = AttendanceDB(db_path)
    db.init_schema()
    try:
        db.upsert_student("S010", "Grace Hopper", section="CSE-A", roll_no="14")
        db.upsert_student("S010", "Grace Hopper Updated")
        row = db.get_student("S010")

        assert row is not None
        assert row["name"] == "Grace Hopper Updated"
        assert row["section"] == "CSE-A"
        assert row["roll_no"] == "14"
    finally:
        db.close()


def test_db_rename_student_preserves_related_rows(tmp_path):
    db_path = tmp_path / "attendance_test.db"
    db = AttendanceDB(db_path)
    db.init_schema()
    try:
        db.upsert_student("S100", "Original Name", section="A", roll_no="21")
        emb = l2_normalize(np.random.rand(512).astype(np.float32))
        db.add_embedding("S100", emb, source_path="sample.jpg")
        db.set_prototype("S100", emb)
        session_id = db.create_session(mode="live", source="camera:0")
        db.mark_attendance(session_id, "S100", confidence=0.91, frame_index=7)

        renamed = db.rename_student(
            old_student_id="S100",
            new_student_id="S101",
            name="Updated Name",
            section="B",
            roll_no="22",
        )

        assert renamed is True
        assert db.get_student("S100") is None

        new_row = db.get_student("S101")
        assert new_row is not None
        assert new_row["name"] == "Updated Name"
        assert new_row["section"] == "B"
        assert new_row["roll_no"] == "22"

        rows = db.get_session_records(session_id)
        assert len(rows) == 1
        assert rows[0]["student_id"] == "S101"
        assert rows[0]["name"] == "Updated Name"

        student_ids, names, matrix = db.load_prototypes()
        assert student_ids == ["S101"]
        assert names == ["Updated Name"]
        assert matrix.shape == (1, 512)
    finally:
        db.close()
